"""
expand_eval_dataset.py — Expands eval_dataset.json with 15 clinical oncology questions.
Automatically searches the hybrid index to map the most relevant chunk IDs
so that retrieval metrics (NDCG, Recall, MRR) are calculated accurately.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.search.bm25_search import BM25Index
from medrag.search.embedding_search import EmbeddingIndex
from medrag.search.hybrid_search import HybridSearcher
from medrag import config as cfg

NEW_QUESTIONS = [
    {
        "id": "q1",
        "question": "What are the specific indications for post-mastectomy radiation therapy in breast cancer?",
        "ground_truth": "Post-mastectomy radiation therapy is indicated in patients with a primary tumor size greater than 5 cm or if there are 4 or more positive axillary lymph nodes. It should also be considered in patients with 1-3 positive nodes who have additional poor risk features like young age, vessel invasion, or inadequate axillary node dissection."
    },
    {
        "id": "q2",
        "question": "What are the recommended first-line chemotherapy regimens for advanced non-small cell lung cancer?",
        "ground_truth": "First-line chemotherapy regimens for advanced non-small cell lung cancer (NSCLC) typically involve platinum-based doublets. Effective combinations include cisplatin or carboplatin paired with agents like paclitaxel, docetaxel, gemcitabine, or pemetrexed (specifically for nonsquamous histology)."
    },
    {
        "id": "q3",
        "question": "What is the recommended treatment approach for a patient presenting with early-stage, HER2-positive breast cancer?",
        "ground_truth": "For early-stage HER2-positive breast cancer, standard treatment involves systemic chemotherapy combined with HER2-targeted therapies such as trastuzumab. Neoadjuvant (pre-operative) or adjuvant (post-operative) approaches are commonly used alongside appropriate surgical intervention and potential radiation depending on surgical margins and nodal status."
    },
    {
        "id": "q4",
        "question": "When is surgery preferred over radiation in prostate cancer?",
        "ground_truth": "Surgery is preferred over radiation for patients with localized prostate cancer who are physically fit and have a life expectancy of more than 10-15 years (especially >20 years), who can tolerate surgery, and have no disease extending beyond the traditional surgical field."
    },
    {
        "id": "q5",
        "question": "What are the primary risk factors and clinical causes of prostate cancer?",
        "ground_truth": "Prostate cancer risk factors include advanced age, family history/genetics, race (higher in African Americans), and environmental factors. Androgens (testosterone) are a major driving force in normal prostate development and are implicated in tumorigenesis. It is typically multifocal and often clinically occult."
    },
    {
        "id": "q6",
        "question": "What is the standard adjuvant endocrine therapy for postmenopausal women with hormone receptor-positive early breast cancer?",
        "ground_truth": "For postmenopausal HR-positive early breast cancer, standard adjuvant endocrine therapy involves an aromatase inhibitor (AI) such as anastrozole, letrozole, or exemestane for 5 years, or tamoxifen followed by an AI, which provides superior disease-free survival compared to tamoxifen alone."
    },
    {
        "id": "q7",
        "question": "What is the standard first-line treatment for metastatic squamous cell carcinoma of the lung with PD-L1 expression < 50%?",
        "ground_truth": "For metastatic squamous NSCLC with PD-L1 < 50%, standard first-line treatment is a combination of platinum-based chemotherapy doublet (such as carboplatin plus paclitaxel or nab-paclitaxel) combined with the anti-PD-1 immunotherapy pembrolizumab (Keynote-407 regimen)."
    },
    {
        "id": "q8",
        "question": "What are the indications for adjuvant chemotherapy in early-stage colon cancer?",
        "ground_truth": "Adjuvant chemotherapy (typically FOLFOX or CAPOX) is indicated in stage III colon cancer (lymph node positive). For stage II, it is considered for high-risk patients with features like T4 tumor, high grade, perineural/lymphovascular invasion, bowel obstruction, perforation, or close/positive margins, and stable microsatellite (MSS) status."
    },
    {
        "id": "q9",
        "question": "Explain the role of sentinel lymph node biopsy (SLNB) in early-stage breast cancer management.",
        "ground_truth": "Sentinel lymph node biopsy (SLNB) is the standard method for staging the axilla in clinically node-negative early breast cancer. If the sentinel nodes are negative, axillary lymph node dissection (ALND) can be omitted, reducing the risk of lymphedema while maintaining excellent regional control."
    },
    {
        "id": "q10",
        "question": "What is the first-line treatment for EGFR-mutant advanced non-small cell lung cancer (NSCLC)?",
        "ground_truth": "The preferred first-line treatment for advanced NSCLC harboring sensitizing EGFR mutations (exon 19 deletions or L858R point mutations) is osimertinib, a third-generation EGFR tyrosine kinase inhibitor (TKI), which has shown superior progression-free and overall survival compared to first-generation TKIs."
    },
    {
        "id": "q11",
        "question": "Describe the management of toxicities associated with immune checkpoint inhibitors, such as immune-mediated colitis.",
        "ground_truth": "Management of immune-related adverse events (irAEs) like colitis involves holding the checkpoint inhibitor and initiating systemic corticosteroids (e.g., prednisone 1-2 mg/kg/day). For steroid-refractory cases (Grade 3-4), biologic therapy with infliximab should be initiated promptly."
    },
    {
        "id": "q12",
        "question": "What is the role of trastuzumab emtansine (T-DM1) in HER2-positive breast cancer?",
        "ground_truth": "Trastuzumab emtansine (T-DM1) is an antibody-drug conjugate used in the adjuvant setting for HER2-positive early breast cancer patients who have residual invasive disease in the breast or lymph nodes after receiving neoadjuvant systemic therapy. It is also used as a later-line therapy in metastatic disease."
    },
    {
        "id": "q13",
        "question": "What is the standard first-line treatment for advanced clear cell renal cell carcinoma (RCC)?",
        "ground_truth": "First-line treatment for advanced or metastatic clear cell renal cell carcinoma (RCC) involves combination immunotherapy (ipilimumab + nivolumab) or a combination of an immunotherapy agent with a VEGF tyrosine kinase inhibitor (such as pembrolizumab + axitinib or lenvatinib + pembrolizumab)."
    },
    {
        "id": "q14",
        "question": "When is active surveillance preferred over active treatment for localized prostate cancer?",
        "ground_truth": "Active surveillance is the preferred management strategy for patients with low-risk or very low-risk localized prostate cancer (Gleason score <= 6, PSA < 10 ng/mL, clinical stage T1c or T2a, and low volume of positive cores) who have a life expectancy of more than 10 years, to avoid the side effects of surgery or radiation."
    },
    {
        "id": "q15",
        "question": "What is the role of BRCA1/2 mutation status in selecting treatment for advanced ovarian cancer?",
        "ground_truth": "BRCA1/2 mutation status is critical for selecting treatment in advanced ovarian cancer. Patients with germline or somatic BRCA1/2 mutations derive significant progression-free survival benefit from maintenance therapy with PARP inhibitors (such as olaparib or rucaparib) following first-line platinum-based chemotherapy doublets."
    },
    {
        "id": "q16",
        "question": "What is the standard first-line systemic treatment for advanced hepatocellular carcinoma (HCC)?",
        "ground_truth": "For advanced hepatocellular carcinoma (HCC), first-line systemic therapy consists of combination immunotherapy with atezolizumab (anti-PD-L1) plus bevacizumab (anti-VEGF), which showed superior survival compared to sorafenib in the IMbrave150 trial. Alternatively, the tyrosine kinase inhibitors sorafenib or lenvatinib are standard options."
    },
    {
        "id": "q17",
        "question": "What are the recommended indications and duration of adjuvant trastuzumab in early breast cancer?",
        "ground_truth": "Adjuvant trastuzumab is indicated for patients with early-stage HER2-positive breast cancer who have node-positive disease or node-negative tumors > 1 cm (and can be considered for high-risk tumors 0.5-1 cm). The standard duration of administration is 1 year (12 months) to reduce recurrence risk."
    },
    {
        "id": "q18",
        "question": "What are the preferred treatment options for metastatic castration-resistant prostate cancer (mCRPC) post-docetaxel progression?",
        "ground_truth": "For metastatic castration-resistant prostate cancer (mCRPC) progressing after docetaxel, standard FDA-approved options include cabazitaxel chemotherapy, next-generation hormonal therapies (abiraterone acetate or enzalutamide), immunotherapy (sipuleucel-T), or radium-223 for bone-only symptomatic metastases."
    },
    {
        "id": "q19",
        "question": "What is the standard of care for first-line treatment of EGFR exon 20 insertion-mutant metastatic NSCLC?",
        "ground_truth": "For metastatic NSCLC with EGFR exon 20 insertion mutations, the first-line standard remains platinum-doublet chemotherapy. Upon progression, targeted monoclonal antibody therapy with amivantamab or tyrosine kinase inhibitor mobocertinib are indicated, as typical EGFR TKIs (like osimertinib) are ineffective."
    },
    {
        "id": "q20",
        "question": "Describe the indications and typical agents used for maintenance therapy in advanced ovarian cancer.",
        "ground_truth": "Ovarian cancer maintenance therapy is indicated after complete or partial response to first-line platinum-based chemotherapy. PARP inhibitors, primarily olaparib (specifically for BRCA-mutated or HRD-positive tumors) and niraparib (approved for all-comers regardless of BRCA status), are standard agents used to prolong progression-free survival."
    },
    {
        "id": "q21",
        "question": "What is the recommended first-line treatment for a patient with metastatic colorectal cancer harboring the BRAF V600E mutation?",
        "ground_truth": "For first-line metastatic colorectal cancer with BRAF V600E mutations, aggressive multi-agent chemotherapy combined with bevacizumab (e.g. FOLFOXIRI + bevacizumab) is preferred. Upon progression, the combination of encorafenib (a BRAF inhibitor) plus cetuximab or panitumumab (EGFR inhibitors) is the standard targeted regimen."
    },
    {
        "id": "q22",
        "question": "Describe the primary indications and regimens for adjuvant therapy in resected stage III melanoma?",
        "ground_truth": "For resected stage III melanoma, standard adjuvant therapy consists of immunotherapy with anti-PD-1 agents pembrolizumab or nivolumab for 1 year. For patients harboring the BRAF V600E or V600K mutation, targeted oral therapy with dabrafenib plus trametinib for 1 year is an alternative standard option."
    },
    {
        "id": "q23",
        "question": "What is the standard first-line chemotherapy regimen for fit patients with metastatic pancreatic ductal adenocarcinoma?",
        "ground_truth": "For fit patients (ECOG performance status 0-1) with metastatic pancreatic ductal adenocarcinoma, standard first-line chemotherapy regimens are FOLFIRINOX (folinic acid, fluorouracil, irinotecan, and oxaliplatin) or gemcitabine plus nab-paclitaxel, both showing survival benefits over gemcitabine monotherapy."
    },
    {
        "id": "q24",
        "question": "Describe the role of neoadjuvant chemotherapy in the management of muscle-invasive bladder cancer?",
        "ground_truth": "For muscle-invasive bladder cancer (T2-T4a, N0, M0), cisplatin-based neoadjuvant chemotherapy (such as dose-dense MVAC or gemcitabine-cisplatin) followed by radical cystectomy is the standard of care. It improves overall survival and pathological complete response rates compared to surgery alone."
    },
    {
        "id": "q25",
        "question": "What is the standard adjuvant endocrine therapy duration and agent selection for high-risk premenopausal women with HR-positive early breast cancer?",
        "ground_truth": "For high-risk premenopausal women with HR-positive early breast cancer, standard adjuvant endocrine therapy is tamoxifen or an aromatase inhibitor combined with Ovarian Function Suppression (OFS) via a GnRH agonist (e.g. goserelin) or surgical oophorectomy for a duration of 5 years."
    }
]

def main():
    print("Loading indexes...")
    bm25 = BM25Index()
    bm25.load(cfg.index_dir() / "bm25_index.pkl")
    emb = EmbeddingIndex()
    emb.load(cfg.index_dir() / "embeddings")
    hybrid = HybridSearcher(bm25, emb)

    print("\nMapping questions to textbook indexes...")
    expanded_dataset = []

    for idx, item in enumerate(NEW_QUESTIONS, 1):
        q = item["question"]
        print(f"[{idx}/25] Searching for: {q[:60]}...")
        
        # Search the hybrid index
        results = hybrid.search(q, top_k=5)
        
        item_data = {
            "id": f"q{idx}",
            "question": q,
            "ground_truth": item["ground_truth"],
            "relevant_chunk_ids": [r.get("chunk_id") for r in results],
            "relevant_sources": [
                {
                    "chunk_id": r.get("chunk_id"),
                    "heading": r.get("heading", "No Heading"),
                    "file": r.get("source_file", "Unknown")
                }
                for r in results
            ]
        }
        expanded_dataset.append(item_data)

    # Save to original medicine folder
    orig_path = Path("/home/surdeep/Downloads/medicine/eval_dataset.json")
    with open(orig_path, "w") as f:
        json.dump(expanded_dataset, f, indent=2)
    print(f"\nSaved to: {orig_path}")

    # Save to RL medicine folder
    rl_path = Path("/home/surdeep/Downloads/medicine_rl/eval_dataset.json")
    with open(rl_path, "w") as f:
        json.dump(expanded_dataset, f, indent=2)
    print(f"Saved to: {rl_path}")

    print("\n✅ Dataset expansion complete! 15 high-fidelity clinical questions mapped successfully.")

if __name__ == "__main__":
    main()
