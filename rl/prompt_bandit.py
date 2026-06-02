"""
prompt_bandit.py — Multi-Armed Bandit for Prompt Template Optimization.

Uses Thompson Sampling (Bayesian approach) to learn which prompt template
produces the highest reward. Each prompt is an "arm" of the bandit.

How it works:
  1. Each prompt template has a Beta distribution: Beta(alpha, beta)
     - alpha = number of "successes" (high rewards)
     - beta  = number of "failures" (low rewards)
  2. To choose a prompt: sample from each Beta distribution, pick highest
  3. After getting a reward: update the chosen prompt's alpha/beta
  4. Over time, the best prompt gets chosen more and more (exploitation)
     while occasionally trying others (exploration)

Thompson Sampling Formula:
  For each arm i:
    theta_i ~ Beta(alpha_i, beta_i)      # sample from posterior
  Choose arm: a = argmax(theta_i)         # pick highest sample
  After reward r:
    alpha_a += r                          # update success count
    beta_a  += (1 - r)                    # update failure count
"""

import json
import random
import numpy as np
from pathlib import Path


# ── Prompt Template Pool ──────────────────────────────────────────────────
# Each template has a different style/instruction for the LLM.
# The bandit will learn which one produces better answers.

PROMPT_TEMPLATES = [
    {
        "id": "clinical_precise",
        "name": "Clinical Precise",
        "system": "You are a board-certified oncologist. Answer the question using ONLY the provided context. Follow these strict rules:\n1. DEMOGRAPHIC SCORING: Never drop subgroup constraints (e.g., 'in women', 'in children', 'in specific stages'). Never generalize subgroup-specific statistics to the general population.\n2. GUIDELINE PRIORITY: If sources present conflicting recommendations, prioritize modern global guidelines (e.g., FIT over FOBT/DRE for colorectal screening) and highlight the modern standard.\n3. TEXTUAL ANCHORING: Cite evidence from the sources accurately without guessing or approximating values.",
        "user_template": "Clinical Context:\n{context}\n\nClinical Question: {query}\n\nProvide a precise, evidence-based answer:"
    },
    {
        "id": "structured_medical",
        "name": "Structured Medical",
        "system": "You are a medical expert assistant. Answer based strictly on the provided context. Follow these rules:\n1. Never drop demographic restrictions (e.g., gender, age group, tumor stage).\n2. Prioritize modern diagnostic and therapeutic standards of care.\n3. Structure your answer with clear, precise key points based on the evidence.",
        "user_template": "Evidence from Medical Literature:\n{context}\n\nQuestion: {query}\n\nAnswer with structured key points based on the evidence above:"
    },
    {
        "id": "concise_direct",
        "name": "Concise Direct",
        "system": "You are a clinical decision support system. Give a direct, concise answer using only the provided medical evidence. Do not add information not in the context.",
        "user_template": "Retrieved Medical Evidence:\n{context}\n\nQuery: {query}\n\nDirect answer:"
    },
    {
        "id": "comprehensive_review",
        "name": "Comprehensive Review",
        "system": "You are a medical literature reviewer. Synthesize the provided context into a comprehensive answer. Reference specific sources and findings.",
        "user_template": "Literature Sources:\n{context}\n\nResearch Question: {query}\n\nComprehensive synthesis of the evidence:"
    },
    {
        "id": "treatment_focused",
        "name": "Treatment Focused",
        "system": "You are an oncology treatment specialist. Focus on treatment recommendations, drug regimens, and clinical outcomes. Use only the provided context.",
        "user_template": "Clinical Evidence:\n{context}\n\nTreatment Question: {query}\n\nTreatment recommendations based on the evidence:"
    },
    {
        "id": "differential_reasoning",
        "name": "Differential Reasoning",
        "system": "You are a clinical reasoning expert. Analyze the provided evidence carefully, consider different aspects, and provide a well-reasoned answer.",
        "user_template": "Medical Evidence Base:\n{context}\n\nClinical Question: {query}\n\nReasoned analysis and answer:"
    },
    {
        "id": "simple_extraction",
        "name": "Simple Extraction",
        "system": "Extract the answer to the question from the provided text. If the answer is not in the text, say so.",
        "user_template": "Text:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    },
    {
        "id": "guideline_based",
        "name": "Guideline Based",
        "system": "You are a clinical guideline interpreter. Answer based on the provided guidelines and textbook evidence. Follow these rules:\n1. DEMOGRAPHIC CONSTRAINTS: Be highly precise regarding subgroups (e.g., 'in women', 'in children'). Never generalize them.\n2. MODERN STANDARD: Prioritize and emphasize the most modern guidelines and screening intervals.\n3. PROTOCOL ANCHORING: Mention specific staging systems and treatment protocols accurately.",
        "user_template": "Clinical Guidelines and Textbook Evidence:\n{context}\n\nGuideline Question: {query}\n\nGuideline-based recommendation:"
    },
]


class PromptBandit:
    """
    Thompson Sampling bandit for prompt template selection.

    State is persisted to disk so learning carries across sessions.
    """

    def __init__(self, state_path: str = "rl/rl_state.json"):
        self.state_path = Path(state_path)
        self.templates = PROMPT_TEMPLATES
        self.n_arms = len(self.templates)

        # Load or initialize Beta distribution parameters
        if self.state_path.exists():
            self._load_state()
        else:
            # Uniform prior: Beta(1, 1) = uniform on [0, 1]
            self.alphas = [1.0] * self.n_arms
            self.betas = [1.0] * self.n_arms
            self.total_pulls = [0] * self.n_arms
            self.total_rewards = [0.0] * self.n_arms
            self.history = []

    def select_arm(self) -> tuple[int, dict]:
        """
        Thompson Sampling: sample from each arm's Beta posterior,
        return the arm with the highest sample.

        Returns:
            (arm_index, template_dict)
        """
        # Sample from Beta(alpha, beta) for each arm
        samples = [
            np.random.beta(self.alphas[i], self.betas[i])
            for i in range(self.n_arms)
        ]
        chosen = int(np.argmax(samples))
        return chosen, self.templates[chosen]

    def update(self, arm_index: int, reward: float):
        """
        Update the Beta posterior for the chosen arm.

        Thompson Sampling update:
          alpha += reward       (more successes shift distribution right)
          beta  += (1 - reward) (more failures shift distribution left)

        Args:
            arm_index: Which prompt was used
            reward:    Scalar reward in [0, 1]
        """
        reward = max(0.0, min(1.0, reward))  # clamp to [0, 1]

        self.alphas[arm_index] += reward
        self.betas[arm_index] += (1.0 - reward)
        self.total_pulls[arm_index] += 1
        self.total_rewards[arm_index] += reward

        self.history.append({
            "arm": arm_index,
            "template": self.templates[arm_index]["id"],
            "reward": round(reward, 4),
        })

        self._save_state()

    def get_best_arm(self) -> tuple[int, dict]:
        """Return the arm with the highest expected reward (exploitation only)."""
        expected = [
            self.alphas[i] / (self.alphas[i] + self.betas[i])
            for i in range(self.n_arms)
        ]
        best = int(np.argmax(expected))
        return best, self.templates[best]

    def get_stats(self) -> list[dict]:
        """Return stats for all arms, sorted by expected reward."""
        stats = []
        for i in range(self.n_arms):
            expected = self.alphas[i] / (self.alphas[i] + self.betas[i])
            stats.append({
                "arm": i,
                "id": self.templates[i]["id"],
                "name": self.templates[i]["name"],
                "expected_reward": round(expected, 4),
                "pulls": self.total_pulls[i],
                "avg_reward": round(self.total_rewards[i] / max(1, self.total_pulls[i]), 4),
                "alpha": round(self.alphas[i], 2),
                "beta": round(self.betas[i], 2),
            })
        stats.sort(key=lambda x: x["expected_reward"], reverse=True)
        return stats

    def _save_state(self):
        """Persist bandit state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "alphas": self.alphas,
            "betas": self.betas,
            "total_pulls": self.total_pulls,
            "total_rewards": self.total_rewards,
            "history": self.history[-200:],  # keep last 200 entries
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load bandit state from disk."""
        with open(self.state_path) as f:
            state = json.load(f)
        self.alphas = state["alphas"]
        self.betas = state["betas"]
        self.total_pulls = state["total_pulls"]
        self.total_rewards = state["total_rewards"]
        self.history = state.get("history", [])

        # Ensure arrays match template count (in case templates were added)
        while len(self.alphas) < self.n_arms:
            self.alphas.append(1.0)
            self.betas.append(1.0)
            self.total_pulls.append(0)
            self.total_rewards.append(0.0)
