"""
train_rl.py — Main RL Training Loop for Medical RAG.

This script runs the RL training loop:
  1. Load eval dataset (questions + ground truth answers)
  2. For each episode:
     a. Prompt Bandit selects a prompt template
     b. Retrieval Optimizer suggests search parameters
     c. Run the RAG pipeline with those settings
     d. Compute reward from the output quality
     e. Update both the bandit and the optimizer
  3. After all episodes, print the learned best configuration

Usage:
  cd /home/surdeep/Downloads/medicine_rl
  python3 rl/train_rl.py --episodes 30
"""

import sys
import json
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.prompt_bandit import PromptBandit
from rl.retrieval_optimizer import RetrievalOptimizer
from rl.rewards import RewardComputer


def load_eval_dataset(path: str = "eval_dataset.json") -> list:
    """Load evaluation questions and ground truth answers."""
    with open(path) as f:
        return json.load(f)


def build_pipeline():
    """Build the RAG pipeline (lazy import to avoid slow startup)."""
    from medrag.hm_rag import HMRAGPipeline
    return HMRAGPipeline()


def run_episode(pipeline, bandit: PromptBandit, retrieval_opt: RetrievalOptimizer,
                reward_computer: RewardComputer, question: str, ground_truth: str,
                episode_num: int):
    """
    Run one RL episode:
      1. Select prompt template (bandit)
      2. Select retrieval params (Bayesian opt)
      3. Run pipeline
      4. Compute reward
      5. Update both agents
    """
    # ── Action Selection ──────────────────────────────────────────────
    arm_idx, template = bandit.select_arm()
    ret_params = retrieval_opt.suggest_params()

    print(f"\n{'─'*60}")
    print(f"Episode {episode_num}")
    print(f"  Prompt:  {template['name']} (arm {arm_idx})")
    print(f"  Params:  alpha={ret_params['alpha']}, top_k={ret_params['top_k']}, "
          f"rrf_k={ret_params['rrf_k']}, reranker_top_n={ret_params['reranker_top_n']}")
    print(f"  Query:   {question}")
    print(f"  [RAG Stage] Starting execution...")

    # ── Environment Step (run pipeline) ───────────────────────────────
    start = time.time()

    # Progress callback to show stage updates live
    def print_stage(msg):
        print(f"  [RAG Stage] {msg}")

    # Apply retrieval params to pipeline
    pipeline.hybrid.alpha = ret_params["alpha"]
    pipeline.hybrid.rrf_k = ret_params["rrf_k"]

    # Build the custom prompt using the bandit's selected template
    result = pipeline.run(
        question,
        progress_callback=print_stage,
        prompt_override={
            "system": template["system"],
            "user_template": template["user_template"],
        },
        top_k=ret_params["top_k"],
        reranker_top_n=ret_params["reranker_top_n"],
    )

    latency = time.time() - start
    answer = result["answer"]
    context = result["context"]

    # ── Reward Computation ────────────────────────────────────────────
    reward_info = reward_computer.compute_reward(
        answer=answer,
        ground_truth=ground_truth,
        context_chunks=context,
        query=question,
        latency=latency,
    )
    reward = reward_info["total_reward"]

    # ── Policy Update ─────────────────────────────────────────────────
    bandit.update(arm_idx, reward)
    retrieval_opt.report_reward(ret_params, reward)

    # ── Logging ───────────────────────────────────────────────────────
    print(f"\n  Answer Generated:\n{answer}\n")
    print(f"  Reward:  {reward:.4f}  (SBERT={reward_info['sbert_sim']:.3f}, "
          f"Faith={reward_info['faithfulness']:.3f}, ROUGE={reward_info['rouge_l']:.3f})")
    print(f"  Latency: {latency:.1f}s")

    return reward_info


def main():
    parser = argparse.ArgumentParser(description="RL Training for Medical RAG")
    parser.add_argument("--episodes", type=int, default=24,
                        help="Number of training episodes (default: 24 = 8 prompts × 3 questions)")
    parser.add_argument("--eval-data", type=str, default="eval_dataset.json",
                        help="Path to evaluation dataset")
    args = parser.parse_args()

    print("=" * 60)
    print("  MEDICAL RAG — RL TRAINING")
    print("  Prompt Optimization + Retrieval Parameter Tuning")
    print("=" * 60)

    # ── Initialize Components ─────────────────────────────────────────
    print("\nInitializing pipeline and RL agents...")
    pipeline = build_pipeline()
    bandit = PromptBandit(state_path="rl/rl_state.json")
    retrieval_opt = RetrievalOptimizer(state_path="rl/retrieval_state.json")
    reward_computer = RewardComputer(sbert_model=pipeline.sbert_model)

    dataset = load_eval_dataset(args.eval_data)
    print(f"Loaded {len(dataset)} evaluation questions")
    print(f"Running {args.episodes} episodes")

    # ── Training Loop ─────────────────────────────────────────────────
    all_rewards = []

    for ep in range(args.episodes):
        # Cycle through questions
        item = dataset[ep % len(dataset)]
        question = item["question"]
        ground_truth = item["ground_truth"]

        reward_info = run_episode(
            pipeline, bandit, retrieval_opt, reward_computer,
            question, ground_truth, ep + 1
        )
        all_rewards.append(reward_info["total_reward"])

    # ── Results ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RL TRAINING COMPLETE")
    print("=" * 60)

    # Prompt Bandit Results
    print("\n📋 Prompt Template Rankings (by expected reward):")
    for stat in bandit.get_stats():
        bar = "█" * int(stat["expected_reward"] * 20)
        print(f"  {stat['name']:<25} {bar} {stat['expected_reward']:.4f}  "
              f"(pulls={stat['pulls']}, avg_reward={stat['avg_reward']:.4f})")

    best_arm_idx, best_template = bandit.get_best_arm()
    print(f"\n🏆 Best Prompt: {best_template['name']}")

    # Retrieval Optimizer Results
    ret_stats = retrieval_opt.get_stats()
    print(f"\n🔍 Best Retrieval Parameters (after {ret_stats['n_observations']} trials):")
    for k, v in ret_stats["best_params"].items():
        print(f"  {k}: {v}")
    print(f"  Best reward: {ret_stats['best_reward']:.4f}")

    # Reward Trend
    if len(all_rewards) >= 4:
        first_half = sum(all_rewards[:len(all_rewards)//2]) / (len(all_rewards)//2)
        second_half = sum(all_rewards[len(all_rewards)//2:]) / (len(all_rewards) - len(all_rewards)//2)
        improvement = ((second_half - first_half) / first_half) * 100 if first_half > 0 else 0
        print(f"\n📈 Reward Trend:")
        print(f"  First half avg:  {first_half:.4f}")
        print(f"  Second half avg: {second_half:.4f}")
        print(f"  Improvement:     {improvement:+.1f}%")

    print(f"\n✅ RL state saved to rl/rl_state.json and rl/retrieval_state.json")
    print(f"   These will be loaded automatically next time for continued learning.")


if __name__ == "__main__":
    main()
