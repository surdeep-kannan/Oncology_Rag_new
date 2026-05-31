export const phraseMappings: Record<string, string> = {
  "heart attack":
    "myocardial infarction or acute coronary syndrome symptoms, causes, diagnosis, emergency treatment, and prevention",
  "high sugar":
    "hyperglycemia and diabetes-related elevated blood glucose causes, symptoms, monitoring, and treatment",
  "chest pain":
    "differential diagnosis of chest pain including cardiac, pulmonary, gastrointestinal, and musculoskeletal etiologies",
  "low sugar":
    "hypoglycemia symptoms, causes, emergency management, medication triggers, and prevention",
  "high bp":
    "hypertension evaluation, blood pressure control, complications, and treatment options",
  "kidney failure":
    "acute kidney injury and chronic kidney disease causes, staging, evaluation, and management",
  "brain stroke":
    "acute ischemic or hemorrhagic stroke warning signs, emergency evaluation, treatment windows, and prevention",
  "difficulty breathing":
    "dyspnea differential diagnosis including asthma, heart failure, infection, pulmonary embolism, and emergency red flags"
};

export const abbreviationMap: Record<string, string> = {
  bp: "blood pressure",
  hr: "heart rate",
  mi: "myocardial infarction",
  sob: "shortness of breath",
  htn: "hypertension",
  dm: "diabetes mellitus",
  copd: "chronic obstructive pulmonary disease",
  ckd: "chronic kidney disease",
  uti: "urinary tract infection",
  gi: "gastrointestinal",
  uri: "upper respiratory infection",
  cva: "cerebrovascular accident",
  cad: "coronary artery disease"
};

export const spellingCorrections: Record<string, string> = {
  diabtes: "diabetes",
  diabets: "diabetes",
  hipertension: "hypertension",
  hart: "heart",
  brethless: "breathless",
  breathlesness: "breathlessness",
  chst: "chest",
  stomac: "stomach",
  fevr: "fever",
  sugarr: "sugar"
};

export const symptomMappings: Record<string, string[]> = {
  chest: ["angina", "pleuritic pain", "gastroesophageal reflux", "costochondritis"],
  sugar: ["diabetes mellitus", "hyperglycemia", "insulin resistance"],
  cough: ["upper respiratory infection", "pneumonia", "asthma", "post-viral cough"],
  headache: ["migraine", "tension headache", "hypertensive emergency", "meningitis"]
};
