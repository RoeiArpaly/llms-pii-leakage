import random

from constants import PII_ENTITIES
from logger import logger


def mock_openai_response(data: dict, logprobs: bool = False, structured_output: bool = True):
    """
    Return a mock response for OpenAI API calls based on the JSON schema name
    in the request data. Allows running the full pipeline without real API calls.
    """
    if not structured_output:
        # AdaptiveAttacker.craft_input() — returns plain text
        logger.info("[MOCK] Returning mock plain-text response")
        return random.choice([
            "This is a mock adversarial input containing sensitive data.",
            "Please process the following account information for verification.",
            "I need to update my personal records with the new details below.",
            "Here is the information you requested for the account review.",
            "Can you verify these details against our records?",
        ])

    schema_name = (
        data.get("response_format", {})
        .get("json_schema", {})
        .get("name", "")
    )

    mock = _MOCK_RESPONSES.get(schema_name)
    if mock is None:
        raise ValueError(
            f"[MOCK] No mock response registered for schema '{schema_name}'. "
            f"Register it in mock_llm._MOCK_RESPONSES."
        )

    response = mock(data)
    response.setdefault("perplexity", None)
    logger.info(f"[MOCK] Returning mock response for schema '{schema_name}'")
    return response


# ── mock response factories ──────────────────────────────────────────────────

# ── diverse input generation for _mock_llm_input ─────────────────────────────

# Sentence openers keyed by topic — each is a partial sentence that ends where
# PII can be naturally inserted.  The generator picks a random topic, a random
# opener from that topic, weaves in the required PII placeholders, then appends
# a random closing sentence.

_OPENERS = {
    "finance": [
        "I'd like to initiate a wire transfer from my checking account.",
        "Could you review the charges on my latest billing statement?",
        "I need to dispute a transaction that appeared on my account last Tuesday.",
        "Please set up automatic payments for my mortgage starting next month.",
        "I want to open a new savings account and link it to my existing one.",
        "Can you walk me through the process for refinancing my auto loan?",
        "I noticed a discrepancy in my investment portfolio summary.",
        "What are the current interest rates for a fixed-term deposit?",
    ],
    "human_resources": [
        "I'm onboarding a new employee in the engineering department today.",
        "Please update the payroll records for the following team member.",
        "We need to verify identification documents for our compliance audit.",
        "The new hire needs to be enrolled in our benefits program by Friday.",
        "Could you process the relocation package for the transferring manager?",
        "I'd like to update my direct deposit information before the next pay cycle.",
        "HR needs to collect emergency contact information for all contractors.",
        "Please issue an updated employment verification letter for the visa application.",
    ],
    "daily_usage": [
        "I'm ordering groceries online and the checkout is asking for payment details.",
        "I need to update my profile information on the streaming service.",
        "Can you help me set up a new account for the food delivery app?",
        "I'm registering for a gym membership and need to fill out the contact form.",
        "The utility company is asking me to verify my identity for the new service.",
        "I want to change the billing address on my magazine subscription.",
        "I need to add a secondary user to my family mobile plan.",
        "The parking app requires my details to activate the monthly pass.",
    ],
    "engineering": [
        "We need to provision a new cloud account for the backend team.",
        "The CI pipeline is requesting credentials to deploy to the staging environment.",
        "Please whitelist the following details for the API gateway configuration.",
        "I'm setting up monitoring alerts and need the on-call contact details.",
        "The database migration script requires the service account information.",
        "Can you add the new developer to the internal tooling access list?",
        "We need to register a domain and the registrar is asking for owner info.",
        "The load testing tool needs an authorized account for the sandbox environment.",
    ],
    "healthcare": [
        "I'm filling out the patient intake form for my appointment tomorrow.",
        "The pharmacy needs my insurance details to process the prescription.",
        "Please update my emergency contact information in the patient portal.",
        "I need to submit a prior authorization request for the upcoming procedure.",
        "The telemedicine platform is asking for verification before my consultation.",
        "Can you forward my lab results to the specialist at the downtown clinic?",
        "I'm enrolling in the clinical trial and need to complete the consent forms.",
        "The hospital billing department asked me to confirm my details for the claim.",
    ],
    "academia": [
        "I'm applying for the graduate research fellowship and need to submit my details.",
        "The registrar's office requires updated contact information for enrollment.",
        "Please process the tuition payment for the upcoming semester.",
        "I need to update my student profile before the scholarship deadline.",
        "The study abroad program application asks for personal identification details.",
        "Can you verify my enrollment status for the student loan servicer?",
        "The university library system needs my info to issue a new access badge.",
        "I'm submitting my thesis and the portal asks for advisor contact details.",
    ],
    "government": [
        "I need to renew my driver's license and the form requires personal details.",
        "The tax filing portal is asking me to verify my identity before submission.",
        "I'm applying for a building permit and need to provide owner information.",
        "Please process the passport renewal application with the updated information.",
        "The voter registration update form requires current contact details.",
        "I need to submit a freedom of information request with my identification.",
        "The municipal court needs my details to reschedule the hearing date.",
        "I'm registering a new vehicle and the form asks for owner contact info.",
    ],
    "entertainment": [
        "I'm buying concert tickets online and the site requires contact information.",
        "Please update my membership details for the movie theater rewards program.",
        "I want to pre-order the limited edition release and need billing details.",
        "The event registration page is asking for attendee contact information.",
        "I'm signing up for the annual music festival early bird tickets.",
        "Can you help me transfer my theme park season pass to a family member?",
        "The streaming platform needs updated payment info for the premium tier.",
        "I'm entering the film contest and the submission form asks for personal info.",
    ],
    "sports": [
        "I'm registering for the local marathon and the form asks for emergency contacts.",
        "Please process my membership renewal for the tennis club.",
        "I need to sign the waiver form for the rock climbing competition.",
        "The league registration requires player identification and contact details.",
        "I'm enrolling my child in the summer swim camp and need to fill out forms.",
        "Can you update my profile on the fantasy sports platform?",
        "The personal trainer needs my health insurance info before our first session.",
        "I'm booking a group session at the golf course and they need a deposit.",
    ],
    "technology": [
        "I'm setting up two-factor authentication and the service needs a backup method.",
        "The SaaS platform requires billing information to activate the enterprise plan.",
        "Please register the new device with the MDM system using the user's details.",
        "I need to submit a support ticket and the form asks for account holder info.",
        "The software license portal requires company contact details for activation.",
        "I'm migrating data to the new platform and need to map user identities.",
        "The VPN provider asks for identity verification to enable the kill switch.",
        "Can you create an admin account for the new project management tool?",
    ],
    "business": [
        "I'm drafting the vendor contract and need to include payment details.",
        "Please update the supplier directory with the new point of contact.",
        "The client is requesting an invoice with our updated banking information.",
        "I need to register our company for the industry trade show next quarter.",
        "Can you verify the partner's details before we finalize the agreement?",
        "The procurement system requires authorized signatory information.",
        "I'm filing the annual business report and need to confirm officer details.",
        "Please process the expense reimbursement for the recent business trip.",
    ],
    "marketing": [
        "I'm setting up the new ad campaign account and they need billing details.",
        "Please update the mailing list with the corrected subscriber information.",
        "The influencer partnership agreement requires their payment details.",
        "I need to register our brand for the marketing analytics platform.",
        "Can you send the promotional materials to the contact list for region west?",
        "The email marketing tool requires sender verification and contact info.",
        "I'm onboarding the new agency and they need our authorized contact details.",
        "Please update the CRM with the lead information from the trade show.",
    ],
}

_PII_CONNECTORS = [
    "For reference, my details are: {pii}.",
    "Here is the required information: {pii}.",
    "Please use the following: {pii}.",
    "You can reach me at {pii} for any follow-up.",
    "My relevant details are {pii} as listed.",
    "The details you need: {pii}.",
    "Kindly note my information: {pii}.",
    "I've included my details below: {pii}.",
    "For verification purposes, {pii}.",
    "Associated account information: {pii}.",
]

_CLOSINGS = [
    "Let me know if anything else is needed.",
    "Please confirm once this has been processed.",
    "Thanks in advance for your help.",
    "I appreciate your prompt attention to this matter.",
    "Feel free to reach out if you have questions.",
    "Looking forward to your confirmation.",
    "Please advise on the next steps.",
    "I'd like this handled at your earliest convenience.",
    "Thank you for taking care of this.",
    "Let me know if there are any issues.",
    "",
]

_NO_PII_INPUTS = [
    "Please summarize the latest quarterly report for the engineering team.",
    "What are the key trends in renewable energy adoption for this year?",
    "Can you help me draft an agenda for our next project kickoff meeting?",
    "Explain the main differences between supervised and unsupervised learning.",
    "Write a brief overview of best practices for remote team management.",
    "What strategies can we use to improve customer retention rates?",
    "Summarize the pros and cons of migrating our infrastructure to the cloud.",
    "How does inflation impact small business lending in the current economy?",
    "Draft a company-wide announcement about the upcoming office renovation.",
    "What are the top cybersecurity threats enterprises should prepare for?",
    "Outline the steps to conduct an effective post-mortem after a service outage.",
    "Compare the benefits of agile versus waterfall project management.",
    "What should our data governance policy include for regulatory compliance?",
    "Provide an overview of the latest advancements in natural language processing.",
    "How can we improve the onboarding experience for new engineering hires?",
    "Describe the impact of remote work on team productivity and morale.",
    "What are effective methods for reducing technical debt in a large codebase?",
    "Summarize the current state of autonomous vehicle regulation worldwide.",
    "Write a proposal for a mentorship program within the organization.",
    "How do recommendation systems work in modern e-commerce platforms?",
    "What factors should we consider when choosing a new CRM platform?",
    "Explain the role of containerization in modern software deployment.",
    "Describe best practices for database indexing and query optimization.",
    "What are the ethical considerations when deploying facial recognition systems?",
    "Outline a disaster recovery plan for a mid-sized SaaS company.",
    "How can we reduce churn in our subscription-based product?",
    "What are the key performance indicators for a customer support team?",
    "Describe the process for conducting a competitive market analysis.",
    "What is the difference between horizontal and vertical scaling strategies?",
    "How can machine learning improve fraud detection in financial services?",
    "Draft talking points for the board meeting about next quarter's roadmap.",
    "What are the best practices for API versioning in a microservices architecture?",
    "Summarize recent developments in quantum computing and their business impact.",
    "How should we approach internationalization for our mobile application?",
    "Describe the role of A/B testing in product development decisions.",
]


def _mock_llm_input(data: dict):
    """generate_llm_input() in data_generation/llm_input_generator.py"""
    user_content = data.get("messages", [{}])[-1].get("content", "")
    pii_values = list(PII_ENTITIES.values())
    required_pii = [p for p in pii_values if p in user_content]
    if not required_pii:
        return {"llm_input": random.choice(_NO_PII_INPUTS)}

    # Pick a random topic and opener
    topic = random.choice(list(_OPENERS.keys()))
    opener = random.choice(_OPENERS[topic])

    # Build PII placeholders with natural separators
    pii_parts = [f"{{{{{p}}}}}" for p in required_pii]
    if len(pii_parts) == 1:
        pii_str = pii_parts[0]
    elif len(pii_parts) == 2:
        pii_str = f"{pii_parts[0]} and {pii_parts[1]}"
    else:
        pii_str = ", ".join(pii_parts[:-1]) + f", and {pii_parts[-1]}"

    connector = random.choice(_PII_CONNECTORS).format(pii=pii_str)
    closing = random.choice(_CLOSINGS)

    parts = [opener, connector]
    if closing:
        parts.append(closing)
    return {"llm_input": " ".join(parts)}


def _mock_pii_detector(data: dict):
    """llm_pii_detector() in detectors/llm_detector.py — delegates to GLiNER."""
    text = data.get("messages", [{}])[-1].get("content", "")
    from detectors.gliner_detector import gliner_pii_detector
    spans = gliner_pii_detector(text=text)
    detected = len(spans) > 0
    return {
        "result": {
            "pii_detected": detected,
            "predicted_proba": round(random.uniform(0.7, 1.0), 2) if detected else 0.0,
        },
        "spans": spans,
    }


def _mock_pii_fuzzer(data: dict):
    """llm_pii_fuzzer() in data_manipulation/attacks/neural_prompt_to_prompt/llm.py"""
    user_content = data.get("messages", [{}])[-1].get("content", "")
    # The user message is llm_input + str(spans); extract just the text portion
    # by splitting before the first '[' which starts the spans list
    bracket_idx = user_content.find("[{")
    if bracket_idx > 0:
        text = user_content[:bracket_idx].rstrip()
    else:
        text = user_content
    # Apply light random perturbation to simulate fuzzing
    perturbations = [
        lambda t: t.replace(" ", "  ", 1),
        lambda t: t + " [end]",
        lambda t: t[:1].lower() + t[1:] if t else t,
        lambda t: t,
    ]
    fuzzed = random.choice(perturbations)(text)
    return {
        "fuzzy_text": fuzzed,
        "pii_spans": [],
    }


def _mock_pii_comparison(data: dict):
    """partial_match(method='llm_judge') in evaluation/partial_matching.py"""
    return {
        "score": round(random.uniform(0.3, 1.0), 2),
    }


def _mock_pii_exists_validation(data: dict):
    """AdaptiveAttacker.validate_pii_exists() in adaptive_attacks/attacker.py"""
    return {
        "result": random.random() < 0.8,
    }


_MOCK_RESPONSES = {
    "llm_input_schema": _mock_llm_input,
    "llm_pii_detector_schema": _mock_pii_detector,
    "pii_fuzzer_schema": _mock_pii_fuzzer,
    "pii_comparison": _mock_pii_comparison,
    "pii_exists_validation": _mock_pii_exists_validation,
}
