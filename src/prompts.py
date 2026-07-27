import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from config import *
from utils import *

def createJSONExample(cl : str = "", co : str = "10") -> str:
    ret = "{\n" \
        f"\t{quote(synonymClass)}: {quote(cl)},\n" \
        f"\t{quote(confidenceColumn)}: {co}\n" \
        "}"
    
    return ret

def formatInput(label : str = "", definition : str = "", comment : str = "", parents : list = [], children : list = [], synonym : str = "") -> str:
    ret = f"Label: {quote(label)}"

    if addDefinition:
        ret = ret + f"\nDefinition: {quote(definition)}"
    if addComment:
        ret = ret + f"\nComment: {quote(comment)}"
    if addParents:
        ret = ret + f"\nParents: {applyFormat(parents)}"
    if addChildren:
        ret = ret + f"\nChildren: {applyFormat(children)}"
    if len(synonym) > 0:
        ret = ret + f"\nSynonym: {quote(synonym)}"

    return ret

def getExamples() -> str:
    return f"""Examples

Example 1

Input: 
{formatInput("Abnormal talus morphology", "An abnormality of the talus.", "", ["Abnormality of the tarsal bones"], ["Tarsal osteovalgus", "Talar aplasia", "Shortening of the talar neck", "Rocker bottom foot", "Osteolysis of talus", "Osteochondral lesion of talus", "Os trigonum", "Fractured talus", "Delayed talus ossification"], "Abnormal large bone of ankle")}

Output:
{createJSONExample("layperson", "9")}

Example 2

Input:
{formatInput("Menorrhagia", " Prolonged and excessive menses at regular intervals in excess of 80 mL or lasting longer than 7 days.", "", ["Abnormal bleeding", "Abnormality of the menstrual cycle"], [], "Hypermenorrhea")} 

Output:
{createJSONExample("expert", "10")}

Example 3

Input:
{formatInput("High palate", "Height of the palate more than 2 SD above the mean (objective) or palatal height at the level of the first permanent molar more than twice the height of the teeth (subjective).", "The measuring device for this assessment is described in (Hall JG, Froster-Iskenius UG, Allanson JE, Gripp K, Slavotinek A. 2006. Handbook of Normal Physical Measurements. 2nd edition. Oxford Medical, publishers). A high palate is often associated with a narrow palate. However, a narrow palate can easily give a false appearance of a high palate. Height and width of the palate should be assessed and coded separately. We do not recommend the subjective determination because this term can be overused and applied inaccurately.", ["Abnormal palate morphology"], ["High, narrow palate"], "Increased palatal height")} 

Output:
{createJSONExample("expert", "8")}"""

def sourceClassificationPrompt1(
        label : str, 
        definition : str, 
        comment : str,  
        parents : list,
        children : list
) -> str:
    return f"""You are a biomedical ontology expert.

Your first task:

Determine the intended meaning of a Human Phenotype Ontology (HPO) concept. The goal is to establish an accurate understanding of the phenotype.

Instructions:

* Integrate all available concept information to determine the intended phenotype.
* Use the preferred label, definition, comment, parent concepts, and child concepts as complementary sources of information.
* If some fields are missing or incomplete, rely on the remaining information.
* Resolve any ambiguity in the concept's meaning before proceeding.
* Focus only on understanding the phenotype itself. Do not consider how patients or healthcare professionals might describe it.

Provide a concise analysis covering:

* Identify the phenotype described by the HPO concept.
* Determine the scope and specificity of the concept, including what is included or excluded.
* Identify the relevant biomedical context, such as the affected anatomical structure, physiological process, developmental feature, or clinical characteristic.

Context:

{formatInput(label, definition, comment, parents, children, "")}"""

def sourceClassificationPrompt2(
    synonym : str
) -> str:
    return f"""Your second task:
    
Characterize how the given synonym is typically used to refer to the previously established HPO concept.

Discuss:

* Consider the synonym only in the context of the HPO concept established in the previous step.
* Determine who would most commonly choose this exact wording to refer to the phenotype.
* Consider how the synonym would typically be used in real-world communication.
* Consider typical usage in patient education materials, conversations between patients and healthcare providers, clinical documentation, genetics reports, phenotype databases, and biomedical publications.
* Focus on the wording itself rather than the phenotype.
* Base your assessment on contemporary English usage rather than on how technical or plain the wording appears.
* If both healthcare professionals and patients commonly use the synonym, determine which group would account for the majority of its real-world use.

Synonym: {synonym}
"""

def sourceClassificationPrompt3(fewShot : bool = fewShot) -> str:
    ret = f"""Your last task:
    
Using your understanding of the HPO concept and your assessment of the synonym from the previous steps, classify the synonym into one of two categories: "expert" or "layperson"

Your decision should reflect the predominant real-world usage of the synonym when referring to the established HPO concept.

Decision rule:

* Assign the category that best represents the group that would most commonly choose this exact wording.
* If both healthcare professionals and patients commonly use the synonym, choose the category corresponding to the majority of real-world usage.
* Assign a confidence score from 1 to 10 reflecting how certain you are that the assigned category represents the predominant real-world usage.
* Base your decision only on the assessments established in the previous steps.
* Do not reconsider or reinterpret the HPO concept or the synonym.

Output Format (STRICT JSON)

{createJSONExample("'expert' or 'layperson'", '<integer from 1 to 10>')}

Confidence guidelines:

    9–10 = almost certainly correct
    6–8 = likely correct
    3–5 = substantial overlap or ambiguity
    1–2 = highly uncertain"""

    if fewShot:
        ret = ret + "\n\n" + getExamples()

    return ret

def sourceClassificationPrompt(
        label : str, 
        definition : str, 
        comment : str, 
        parents : list, 
        children : list,
        synonym : str,
        fewShot : bool = fewShot
)-> str:
    ret = f"""You are a biomedical ontology expert.

Your task is to classify a synonym of a Human Phenotype Ontology (HPO) concept into one of two categories:

{quote("expert")}: wording that healthcare professionals would typically choose when communicating with other healthcare professionals or documenting phenotypes.
{quote("layperson")}: wording that patients or caregivers would typically choose in everyday communication or patient-facing materials.

The classification concerns the synonym itself, not the underlying phenotype.

Instructions:

    * Use the HPO concept information only to disambiguate the intended meaning of the synonym. Do not base the classification on the wording used elsewhere in the concept description.
    * Imagine that 100 occurrences of the synonym referring to this phenotype were sampled from real-world usage. Assign the category corresponding to the group responsible for the majority of those occurrences.
    * Consider typical usage in patient education materials, conversations between patients and healthcare providers, clinical documentation, genetics reports, phenotype databases, and biomedical publications.
    * Judge the synonym by its typical users rather than by how technical or plain the wording appears.
    * If both groups commonly use the synonym, assign the category that would account for the majority of real-world occurrences.

Output Format (STRICT JSON)

{createJSONExample("'expert' or 'layperson'", '<integer from 1 to 10>')}

Confidence guidelines:

    9–10 = almost certainly correct
    6–8 = likely correct
    3–5 = substantial overlap or ambiguity
    1–2 = highly uncertain

"""

    if fewShot:
        ret = ret + getExamples() + "\n\n"

    ret = ret + f"""Now classify the following input:

{formatInput(label, definition, comment, parents, children, synonym)}"""

    return ret