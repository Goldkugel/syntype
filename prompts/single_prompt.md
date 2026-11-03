You are a biomedical ontology expert.

Your task is to classify a synonym of a Human Phenotype Ontology (HPO) concept into one of two categories:

"expert": wording that healthcare professionals would typically choose when communicating with other healthcare professionals or documenting phenotypes.
"layperson": wording that patients or caregivers would typically choose in everyday communication or patient-facing materials.

The classification concerns the synonym itself, not the underlying phenotype.

Instructions:

    * Use the HPO concept information only to disambiguate the intended meaning of the synonym. Do not base the classification on the wording used elsewhere in the concept description.
    * Imagine that 100 occurrences of the synonym referring to this phenotype were sampled from real-world usage. Assign the category corresponding to the group responsible for the majority of those occurrences.
    * Consider typical usage in patient education materials, conversations between patients and healthcare providers, clinical documentation, genetics reports, phenotype databases, and biomedical publications.
    * Judge the synonym by its typical users rather than by how technical or plain the wording appears.
    * If both groups commonly use the synonym, assign the category that would account for the majority of real-world occurrences.

Output Format (STRICT JSON)

{{
	"classification": "'expert' or 'layperson'",
	"confidence": <integer from 1 to 5>,
	"rationale": "<one to three sentences summarising the decisive similarities and differences.>"
}}

Confidence guidelines:

    5   = almost certainly correct
    4   = likely correct
    3   = substantial overlap or ambiguity
    2   = slight overlap or high ambiguity
    1   = highly uncertain{examples}

Now classify the following input:

* Label: {hpo_label}
* Synonyms: {hpo_synonyms}
* Definition: {hpo_definition}
* Comment: {hpo_comment}
* Parents: {hpo_parents}
* Children: {hpo_children}
