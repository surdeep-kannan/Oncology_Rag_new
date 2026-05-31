import { AlertTriangle, ShieldAlert } from "lucide-react";
import type { PromptOptimizationResult } from "@/types/chat";

export function SafetyBanner({ optimization }: { optimization?: PromptOptimizationResult }) {
  const isEmergency = optimization?.emergency;
  const isUnsafe = optimization?.unsafe;

  return (
    <div className="space-y-3">
      <div className="rounded-3xl border border-amber-500/25 bg-amber-500/10 p-4 text-sm text-amber-100">
        <div className="mb-1 flex items-center gap-2 font-semibold text-amber-50">
          <ShieldAlert className="h-4 w-4" />
          Medical disclaimer
        </div>
        This assistant supports education and workflow acceleration. It does not replace licensed clinical judgment,
        emergency triage, or personalized care.
      </div>

      {(isEmergency || isUnsafe) && (
        <div className="rounded-3xl border border-rose-500/25 bg-rose-500/10 p-4 text-sm text-rose-100">
          <div className="mb-1 flex items-center gap-2 font-semibold text-rose-50">
            <AlertTriangle className="h-4 w-4" />
            Safety escalation
          </div>
          {isEmergency
            ? "Possible emergency symptoms detected. Encourage immediate local emergency evaluation."
            : "Potentially unsafe self-management request detected. Recommend clinician review and risk clarification."}
        </div>
      )}
    </div>
  );
}
