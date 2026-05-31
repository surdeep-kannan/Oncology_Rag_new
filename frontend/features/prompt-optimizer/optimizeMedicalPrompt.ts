import { EMERGENCY_KEYWORDS, UNSAFE_PATTERNS } from "@/lib/constants";
import type { PromptOptimizationResult } from "@/types/chat";
import {
  abbreviationMap,
  phraseMappings,
  spellingCorrections,
  symptomMappings
} from "@/utils/medicalDictionary";

function normalizeWhitespace(input: string) {
  return input.replace(/\s+/g, " ").trim();
}

function correctSpelling(tokens: string[]) {
  const corrections: string[] = [];
  const corrected = tokens.map((token) => {
    const replacement = spellingCorrections[token] ?? token;
    if (replacement !== token) {
      corrections.push(`${token}→${replacement}`);
    }
    return replacement;
  });
  return { corrected, corrections };
}

function expandAbbreviations(tokens: string[]) {
  const detectedMedicalTerms: string[] = [];
  const expanded = tokens.map((token) => {
    const replacement = abbreviationMap[token] ?? token;
    if (replacement !== token) {
      detectedMedicalTerms.push(replacement);
    }
    return replacement;
  });
  return { expanded, detectedMedicalTerms };
}

export function optimizeMedicalPrompt(input: string): PromptOptimizationResult {
  const normalized = normalizeWhitespace(input.toLowerCase());
  const emergency = EMERGENCY_KEYWORDS.some((pattern) => normalized.includes(pattern));
  const unsafe = UNSAFE_PATTERNS.some((pattern) => normalized.includes(pattern));

  if (!normalized) {
    return {
      original: input,
      optimized: "",
      detectedMedicalTerms: [],
      confidence: 0.5,
      corrections: [],
      warnings: ["Enter a symptom, condition, or medical question to begin."],
      unsafe,
      emergency
    };
  }

  if (phraseMappings[normalized]) {
    return {
      original: input,
      optimized: phraseMappings[normalized],
      detectedMedicalTerms: [phraseMappings[normalized].split(" ")[0]],
      confidence: 0.96,
      corrections: [],
      warnings: emergency
        ? ["This may require emergency evaluation. Do not delay urgent care."]
        : [],
      unsafe,
      emergency
    };
  }

  const tokens = normalized.split(" ");
  const { corrected, corrections } = correctSpelling(tokens);
  const { expanded, detectedMedicalTerms } = expandAbbreviations(corrected);

  const keywordEnhancements = expanded.flatMap((token) => symptomMappings[token] ?? []);

  const optimized = normalizeWhitespace(
    [...expanded, ...keywordEnhancements].join(" ").replace(/\s+/g, " ")
  );

  const warnings: string[] = [];
  if (emergency) {
    warnings.push("Possible emergency symptoms detected. Immediate medical care may be appropriate.");
  }
  if (unsafe) {
    warnings.push("The request may be unsafe for self-management without professional guidance.");
  }

  return {
    original: input,
    optimized,
    detectedMedicalTerms: Array.from(new Set([...detectedMedicalTerms, ...keywordEnhancements])).slice(0, 8),
    confidence: corrections.length > 0 ? 0.82 : 0.88,
    corrections,
    warnings,
    unsafe,
    emergency
  };
}
