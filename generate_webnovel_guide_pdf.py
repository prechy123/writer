"""Generate the web-novel writer's guide PDF.

This is a user-facing walkthrough of the two endpoints writers actually
fill in on the platform:

    1. POST /api/profiles/generate/   — build your reusable author voice
    2. POST /api/stories/create/      — start a new book

For every field in each request body we explain:
  - what the field is
  - what the AI actually does with it
  - what to write (and what NOT to write) to get the best result
  - a worked example

Run with:  python generate_webnovel_guide_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT_PATH = Path(__file__).parent / "webnovel_profile_and_story_guide.pdf"


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def build_styles():
    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontSize=24, leading=30,
            spaceAfter=10, textColor=colors.HexColor("#0f172a"),
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontSize=12, leading=16,
            textColor=colors.HexColor("#475569"), spaceAfter=22, alignment=TA_LEFT,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontSize=18, leading=23,
            spaceBefore=18, spaceAfter=10, textColor=colors.HexColor("#0f172a"),
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontSize=14, leading=18,
            spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1e3a8a"),
        ),
        "h3": ParagraphStyle(
            "h3", parent=base["Heading3"], fontSize=11.5, leading=15,
            spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#334155"),
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontSize=10.5, leading=15,
            alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["BodyText"], fontSize=10.5, leading=15,
            leftIndent=14, bulletIndent=4, spaceAfter=2,
        ),
        "callout": ParagraphStyle(
            "callout", parent=base["BodyText"], fontSize=10.5, leading=15,
            leftIndent=10, rightIndent=10, spaceBefore=6, spaceAfter=10,
            textColor=colors.HexColor("#1f2937"),
            backColor=colors.HexColor("#fef9c3"),
            borderColor=colors.HexColor("#eab308"),
            borderWidth=0.5, borderPadding=8,
        ),
        "tip": ParagraphStyle(
            "tip", parent=base["BodyText"], fontSize=10.5, leading=15,
            leftIndent=10, rightIndent=10, spaceBefore=6, spaceAfter=10,
            textColor=colors.HexColor("#064e3b"),
            backColor=colors.HexColor("#ecfdf5"),
            borderColor=colors.HexColor("#10b981"),
            borderWidth=0.5, borderPadding=8,
        ),
        "warn": ParagraphStyle(
            "warn", parent=base["BodyText"], fontSize=10.5, leading=15,
            leftIndent=10, rightIndent=10, spaceBefore=6, spaceAfter=10,
            textColor=colors.HexColor("#7f1d1d"),
            backColor=colors.HexColor("#fef2f2"),
            borderColor=colors.HexColor("#ef4444"),
            borderWidth=0.5, borderPadding=8,
        ),
        "mono": ParagraphStyle(
            "mono", parent=base["BodyText"], fontName="Courier",
            fontSize=9, leading=12, leftIndent=10,
            backColor=colors.HexColor("#f8fafc"),
            borderColor=colors.HexColor("#e2e8f0"),
            borderWidth=0.5, borderPadding=6, spaceAfter=8,
        ),
        "quote": ParagraphStyle(
            "quote", parent=base["BodyText"], fontSize=10.5, leading=15,
            leftIndent=20, rightIndent=20,
            textColor=colors.HexColor("#334155"),
            spaceAfter=8, fontName="Helvetica-Oblique",
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def P(text, style):
    return Paragraph(text, style)


def bullets(items, style):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{t}", style) for t in items]


def field_table(rows):
    """Rows: list of (field, type, required, default)."""
    data = [["Field", "Type", "Required", "Default"]] + rows
    t = Table(data, colWidths=[4.4 * cm, 3.2 * cm, 2.4 * cm, 5.6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f8fafc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


# ---------------------------------------------------------------------------
# Content builders
# ---------------------------------------------------------------------------

def cover(s):
    return [
        P("Web Novel Writer's Guide", s["title"]),
        P(
            "A plain-English walkthrough of the two forms you fill in on the "
            "platform — the Profile form and the Story form — and exactly "
            "what to type in each field so the AI writes in <i>your</i> voice.",
            s["subtitle"],
        ),
        P("How to use this guide", s["h2"]),
        P(
            "On the platform you will interact with two endpoints. Each one "
            "corresponds to a form. This guide walks through every field in "
            "each form, explains what the AI does with that field, and shows "
            "you how to write an answer that gets the best result.",
            s["body"],
        ),
        *bullets([
            "<b>Step 1 — Generate your Profile once.</b> This captures how "
            "<i>you</i> write. You only do this one time.",
            "<b>Step 2 — Create a Story.</b> You can create unlimited stories "
            "that reuse the same profile. Every book will sound like you.",
        ], s["bullet"]),
        Spacer(1, 10),
        P(
            "<b>The golden rule:</b> the AI can only match what you give it. "
            "Vague input makes vague prose. Specific, vivid, honest input "
            "makes specific, vivid, honest prose.",
            s["callout"],
        ),
        PageBreak(),
    ]


def profiles_section(s):
    story = []
    story.append(P("Part 1 — The Profile Form", s["h1"]))
    story.append(P("POST /api/profiles/generate/", s["mono"]))
    story.append(P(
        "This form builds your <b>reusable author voice</b>. When you submit "
        "it, three AI agents (the Profiler, the Empath, and the Masterclass) "
        "read everything you wrote and produce three internal documents:",
        s["body"],
    ))
    story += bullets([
        "<b>Author Profile</b> — how you write (sentences, vocabulary, quirks).",
        "<b>Emotional Guidelines</b> — how you express feelings on the page.",
        "<b>Style Expertise</b> — the craft techniques you lean on.",
    ], s["bullet"])
    story.append(P(
        "Those three documents are then injected into every story you "
        "create using this profile. You do this <b>once</b>, and every book "
        "thereafter sounds like you wrote it.",
        s["body"],
    ))

    story.append(Spacer(1, 4))
    story.append(P("Field overview", s["h2"]))
    story.append(field_table([
        ["name", "string", "Yes", "—"],
        ["background", "string", "No", "(empty)"],
        ["personality", "string", "No", "(empty)"],
        ["communication_style", "string", "No", "(empty)"],
        ["interests_and_values", "string", "No", "(empty)"],
        ["quirks", "string", "No", "(empty)"],
        ["additional_context", "string", "No", "(empty)"],
        ["expert_style", "string", "No", "(empty)"],
        ["expert_writing_sample", "string", "No", "(empty)"],
        ["writing_samples", "list of strings", "No", "[] (empty list)"],
    ]))
    story.append(Spacer(1, 6))
    story.append(P(
        "Technically only <b>name</b> is required. Practically, the more "
        "fields you fill in honestly and concretely, the closer the generated "
        "voice will match yours. Treat every field as an opportunity to teach "
        "the AI something specific about how you write.",
        s["tip"],
    ))

    # -------- name --------
    story.append(P("Field: name", s["h2"]))
    story.append(P(
        "Your pen name or real name. This is a label the AI uses internally "
        "and it will appear in the stored profile. It does not change the "
        "generated prose.",
        s["body"],
    ))
    story.append(P("Example", s["h3"]))
    story.append(P('"name": "Ada Okoro"', s["mono"]))

    # -------- background --------
    story.append(P("Field: background", s["h2"]))
    story.append(P(
        "Who you are as a writer and as a person — where you grew up, the "
        "formative experiences that shaped how you see the world, the "
        "subjects you come back to without meaning to.",
        s["body"],
    ))
    story.append(P("What the AI uses this for", s["h3"]))
    story.append(P(
        "Your background leaks into your prose whether you want it to or not: "
        "the settings you reach for, the conflicts that feel natural to "
        "dramatise, the metaphors you pick. The Profiler agent uses this "
        "field to predict those patterns so the generated prose reads like "
        "it came from someone with your history.",
        s["body"],
    ))
    story.append(P("How to fill it in well", s["h3"]))
    story += bullets([
        "Write 3–6 full sentences. Bullet points are fine.",
        "Be specific: name real places, real jobs, real subcultures.",
        "Mention anything that influenced your taste in fiction.",
        "Don't write a CV — write the parts that changed how you see things.",
    ], s["bullet"])
    story.append(P("Good example", s["h3"]))
    story.append(P(
        '"Grew up in Port Harcourt, eldest of four. Parents ran a small '
        "pharmacy; I spent afternoons behind the counter listening to "
        "customers argue about everything from politics to spoiled yam. "
        "Studied engineering but left to write. I care most about power — "
        "who has it in a room, who pretends not to, and how ordinary people "
        'bend around it."',
        s["quote"],
    ))
    story.append(P("Weak example (avoid)", s["h3"]))
    story.append(P(
        '"I am a writer. I love books and I like to write stories about '
        'life and people."',
        s["warn"],
    ))

    # -------- personality --------
    story.append(P("Field: personality", s["h2"]))
    story.append(P(
        "Your temperament on the page. What actually makes you angry. What "
        "you find funny. What you are afraid of. Whether you lead with "
        "warmth or with distance.",
        s["body"],
    ))
    story.append(P("What the AI uses this for", s["h3"]))
    story.append(P(
        "The Empath agent turns this into emotional guidelines — the rules "
        "that govern how tension rises and falls in your chapters, how your "
        "characters express vulnerability, and what emotional notes you "
        "avoid. Your personality becomes the story's emotional fingerprint.",
        s["body"],
    ))
    story.append(P("How to fill it in well", s["h3"]))
    story += bullets([
        "Pair opposites — e.g. <i>patient with strangers, impatient with family</i>.",
        "Describe what pisses you off. Pettiness welcome. This is gold.",
        "Say whether you deflect with humour, go quiet, or confront head-on.",
        "Name a fear — not a generic one; a specific one.",
    ], s["bullet"])
    story.append(P("Good example", s["h3"]))
    story.append(P(
        '"Dry, impatient, sarcastic with people I love and formally polite '
        "with people I don't. Anger arrives late but lasts. I deflect sadness "
        "with jokes until I can't, and then I get quiet. I hate being "
        "condescended to more than almost anything. Afraid of ending up "
        'boring more than of failing."',
        s["quote"],
    ))

    # -------- communication_style --------
    story.append(P("Field: communication_style", s["h2"]))
    story.append(P(
        "How you actually talk and write when nobody is watching. Short "
        "sentences or long ones. Formal or loose. Do you use semicolons. Do "
        "you swear. Do you pile clauses on top of each other or strip them "
        "down.",
        s["body"],
    ))
    story.append(P("What the AI uses this for", s["h3"]))
    story.append(P(
        "This is the single biggest lever on sentence-level prose. The "
        "Profiler uses it to decide average sentence length, paragraph "
        "rhythm, vocabulary register, dialogue naturalness, and which tics "
        "(em-dashes, fragments, parentheticals) to imitate.",
        s["body"],
    ))
    story.append(P("How to fill it in well", s["h3"]))
    story += bullets([
        "Describe your sentences: short and punchy, long and winding, or mixed.",
        "Name your punctuation habits: em-dashes, ellipses, parentheticals.",
        "Say whether you use slang, pidgin, code-switching, dialect.",
        "Give one-line examples of phrases you actually say or write.",
    ], s["bullet"])
    story.append(P("Good example", s["h3"]))
    story.append(P(
        '"Short sentences, then a long one when I want to slow the reader '
        "down. Heavy em-dash user. I code-switch between English and Pidgin "
        "in dialogue when characters are relaxed. Avoid semicolons. I like "
        "sentence fragments. Often. Contractions always — \"she'd\" not "
        '"she would" unless I\'m being ironic."',
        s["quote"],
    ))

    # -------- interests_and_values --------
    story.append(P("Field: interests_and_values", s["h2"]))
    story.append(P(
        "What you care about deeply, even when it is inconvenient. What you "
        "read about for fun. The hills you will die on. The topics that you "
        "find yourself arguing about at 1am.",
        s["body"],
    ))
    story.append(P("What the AI uses this for", s["h3"]))
    story.append(P(
        "These become the thematic undercurrent of every story. They decide "
        "which details the narrator lingers on, which characters get moral "
        "weight, and what a scene is really <i>about</i> underneath the plot.",
        s["body"],
    ))
    story.append(P("Good example", s["h3"]))
    story.append(P(
        '"Class mobility, and the small humiliations people swallow to keep '
        "moving up. West African folk tales retold without sanitising them. "
        "Architecture and how buildings tell you who a city is for. I value "
        'loyalty, but only the uncomfortable kind — the kind that costs."',
        s["quote"],
    ))

    # -------- quirks --------
    story.append(P("Field: quirks", s["h2"]))
    story.append(P(
        "Your tics. Habits. Catchphrases. Pet peeves. The recurring patterns "
        "your friends could identify even in an anonymous text.",
        s["body"],
    ))
    story.append(P("What the AI uses this for", s["h3"]))
    story.append(P(
        "Quirks are what separates a <i>competent</i> voice replica from one "
        "that actually fools a reader. The Profiler bakes them into "
        "signature moves — recurring images, verbal tics, structural habits "
        "— so the prose has fingerprints.",
        s["body"],
    ))
    story.append(P("How to fill it in well", s["h3"]))
    story += bullets([
        "List actual words and phrases you overuse.",
        "Mention structural habits (e.g. opening chapters on weather).",
        "Include pet peeves — things you <i>refuse</i> to write.",
    ], s["bullet"])
    story.append(P("Good example", s["h3"]))
    story.append(P(
        "\"I overuse the word 'almost'. I end paragraphs on a one-word "
        "sentence when I want a beat. I always describe food before I "
        "describe the room. I refuse to write scenes set in airports — "
        "they bore me. I start a lot of chapters with somebody lying.\"",
        s["quote"],
    ))

    # -------- additional_context --------
    story.append(P("Field: additional_context", s["h2"]))
    story.append(P(
        "Anything else that matters but didn't fit above. A specific author "
        "you are often compared to. A genre you are moving toward or away "
        "from. A creative constraint you want the AI to respect.",
        s["body"],
    ))
    story.append(P("Good example", s["h3"]))
    story.append(P(
        '"I want my prose to feel closer to literary fiction than to '
        "commercial romance, even when the plot is commercial. Avoid purple "
        'prose. Avoid the word "orbs" for eyes. If a sentence sounds like '
        'a movie poster, rewrite it."',
        s["quote"],
    ))

    # -------- writing_samples --------
    story.append(P("Field: writing_samples", s["h2"]))
    story.append(P(
        "The most powerful field on the form. A list of chunks of text "
        "that <i>you</i> actually wrote. Short stories, chapter excerpts, "
        "long posts, even polished messages — anything that is your real "
        "voice on the page.",
        s["body"],
    ))
    story.append(P("What the AI uses this for", s["h3"]))
    story.append(P(
        "All three Phase 1 agents read every sample. The Profiler quotes "
        "short phrases as <i>evidence</i> when building your voice model. "
        "The Empath studies how you handle emotion in context. The "
        "Masterclass studies your craft — pacing, dialogue mechanics, "
        "sensory priorities. Samples are worth more than any number of "
        "bio fields because they show instead of tell.",
        s["body"],
    ))
    story.append(P("How to fill it in well", s["h3"]))
    story += bullets([
        "Provide 2–5 samples. More is better than fewer.",
        "Each sample should be 300–2000 words. Avoid tiny snippets.",
        "Pick samples that represent the voice you want — not old work you "
        "have outgrown.",
        "Include at least one sample with dialogue and one without.",
        "Prefer finished prose over drafts. The AI will imitate quality too.",
        "Do <b>not</b> paste other writers' work. The AI will mimic whatever "
        "you submit.",
    ], s["bullet"])
    story.append(P(
        "If you have to choose between filling in biography fields and "
        "providing writing samples, <b>choose samples every time</b>. A real "
        "paragraph of your prose is worth ten paragraphs of description.",
        s["callout"],
    ))
    story.append(P("Format (JSON)", s["h3"]))
    story.append(P(
        '"writing_samples": [\n'
        '  "Full text of sample 1 ...",\n'
        '  "Full text of sample 2 ...",\n'
        '  "Full text of sample 3 ..."\n'
        "]",
        s["mono"],
    ))

    # -------- expert_style --------
    story.append(P("Fields: expert_style & expert_writing_sample", s["h2"]))
    story.append(P(
        "Optional advanced controls. If you want your prose to also absorb "
        "craft from a specific professional writer you admire, use these "
        "fields. When either is provided, the Masterclass agent uses them "
        "directly instead of inferring style techniques from scratch.",
        s["body"],
    ))
    story.append(P("expert_style", s["h3"]))
    story.append(P(
        "A short essay describing a professional writer's craft — their "
        "pacing, dialogue habits, sensory priorities, structural moves. "
        "Write it in your own words. This is <i>about</i> them, not by them.",
        s["body"],
    ))
    story.append(P("expert_writing_sample", s["h3"]))
    story.append(P(
        "A passage written by that expert, used as craft reference. The AI "
        "studies their technique; it does not copy their voice (yours comes "
        "from the other fields).",
        s["body"],
    ))
    story.append(P(
        "Leave both blank if you are just starting. You can always "
        "regenerate the profile later.",
        s["tip"],
    ))

    # -------- full example --------
    story.append(PageBreak())
    story.append(P("Full request example", s["h2"]))
    story.append(P(
        "Here is what a strong, complete POST body looks like. Notice that "
        "every field is concrete, specific, and short enough to read — no "
        "filler.",
        s["body"],
    ))
    story.append(P(
        'POST /api/profiles/generate/\n'
        'Content-Type: application/json\n\n'
        '{\n'
        '  "name": "Ada Okoro",\n'
        '  "background": "Grew up in Port Harcourt, eldest of four. Parents '
        'ran a pharmacy; afternoons behind the counter taught me how '
        'ordinary people argue about power.",\n'
        '  "personality": "Dry, impatient, sarcastic with people I love. '
        'Anger arrives late and lasts. Deflect sadness with jokes until '
        'I can\'t; then I go quiet.",\n'
        '  "communication_style": "Short sentences, then one long one to '
        'slow the reader. Heavy em-dash user. Code-switch to Pidgin in '
        'relaxed dialogue. Contractions always.",\n'
        '  "interests_and_values": "Class mobility and the small '
        'humiliations it costs. West African folk tales told without '
        'sanitising them. Architecture as politics.",\n'
        '  "quirks": "Overuse \'almost\'. End paragraphs on a one-word '
        'sentence when I want a beat. Describe food before the room. '
        'Often start chapters with someone lying.",\n'
        '  "additional_context": "Literary over commercial even when the '
        'plot is commercial. Avoid purple prose and the word \'orbs\' '
        'for eyes.",\n'
        '  "writing_samples": [\n'
        '    "<paste 500–2000 words of your prose here>",\n'
        '    "<paste another sample here>"\n'
        '  ]\n'
        '}',
        s["mono"],
    ))
    story.append(P("What you get back", s["h3"]))
    story.append(P(
        '{\n'
        '  "profile_id": "b4e9...",\n'
        '  "name": "Ada Okoro",\n'
        '  "author_profile_preview": "...",\n'
        '  "emotional_guidelines_preview": "...",\n'
        '  "expert_styles_preview": "..."\n'
        '}',
        s["mono"],
    ))
    story.append(P(
        "Save the <b>profile_id</b>. You will paste it into every story "
        "request from now on.",
        s["tip"],
    ))

    story.append(PageBreak())
    return story


def stories_section(s):
    story = []
    story.append(P("Part 2 — The Story Form", s["h1"]))
    story.append(P("POST /api/stories/create/", s["mono"]))
    story.append(P(
        "This is the form you fill in every time you start a new book. It "
        "is short on purpose: the heavy lifting (voice, emotional tone, "
        "craft) already lives in your profile. This form tells the AI "
        "<i>which book</i> to write in that voice.",
        s["body"],
    ))

    story.append(P("Field overview", s["h2"]))
    story.append(field_table([
        ["book_title", "string (≤300 chars)", "Yes", "—"],
        ["book_description", "string (≤5000 chars)", "Yes", "—"],
        ["num_chapters", "integer (1–30)", "Yes", "—"],
        ["platform_genre", "string", "No", "(empty)"],
        ["lead_type", "string", "No", "(empty)"],
        ["target_reader", "string", "No", "(empty)"],
        ["content_rating", "string", "No", "(empty)"],
        ["tags", "list of strings", "No", "[]"],
        ["must_include_tropes", "list of strings", "No", "[]"],
        ["must_avoid_tropes", "list of strings", "No", "[]"],
        ["release_goal", "string", "No", "(empty)"],
        ["profile_id", "string", "No", "null"],
        ["initial_chapters", "integer (1–30)", "No", "3"],
    ]))

    story.append(P("Serverless AI model routing (admin setup)", s["h2"]))
    story.append(P(
        "The platform can use different Together serverless models for "
        "different agents. This keeps prose quality high where readers notice "
        "it, while cheaper models handle review, summaries, continuity, and "
        "publishing metadata.",
        s["body"],
    ))
    story.append(P(
        "Recommended environment configuration:\n\n"
        "TOGETHER_DEFAULT_MODEL=MiniMaxAI/MiniMax-M2.7\n"
        "TOGETHER_PLANNER_MODEL=moonshotai/Kimi-K2.5\n"
        "TOGETHER_WRITER_MODEL=moonshotai/Kimi-K2.5\n"
        "TOGETHER_REVIEWER_MODEL=openai/gpt-oss-120b\n"
        "TOGETHER_CONTINUITY_MODEL=MiniMaxAI/MiniMax-M2.7\n"
        "TOGETHER_SUMMARY_MODEL=MiniMaxAI/MiniMax-M2.7\n"
        "TOGETHER_PUBLISHER_MODEL=MiniMaxAI/MiniMax-M2.7",
        s["mono"],
    ))
    story.append(P(
        "If only the legacy TOGETHER_MODEL variable is set, unconfigured roles "
        "fall back to it. This keeps old deployments working while allowing "
        "new deployments to route each pipeline stage deliberately.",
        s["tip"],
    ))

    # -------- book_title --------
    story.append(P("Field: book_title", s["h2"]))
    story.append(P(
        "The working title of the book. Up to 300 characters. This appears "
        "in the manuscript, the dashboard, and is passed to every agent in "
        "the pipeline.",
        s["body"],
    ))
    story.append(P("What the AI uses this for", s["h3"]))
    story.append(P(
        "The Storyteller agent treats the title as a genre and tone signal. "
        "<i>The Girl Who Burned Lagos</i> tells the AI something very "
        "different than <i>Payroll</i>. A vague title like <i>My Book</i> "
        "gives the AI nothing to work with.",
        s["body"],
    ))
    story.append(P("How to fill it in well", s["h3"]))
    story += bullets([
        "Give the title you would actually put on the cover, even if it is "
        "provisional.",
        "Make it carry weight — a concrete image, a concrete tension, a "
        "character name, or a promise.",
        "If you are unsure, write the title that matches the <i>feeling</i> "
        "you want the prose to have.",
    ], s["bullet"])
    story.append(P("Good vs weak titles", s["h3"]))
    story.append(P(
        '✓ "The Girl Who Burned Lagos"<br/>'
        '✓ "Payroll"<br/>'
        '✓ "Things We Do Not Bury"<br/>'
        '✗ "My Novel"<br/>'
        '✗ "Book Idea 3"<br/>'
        '✗ "Untitled"',
        s["body"],
    ))

    # -------- book_description --------
    story.append(P("Field: book_description", s["h2"]))
    story.append(P(
        "By far the most important field on this form. Up to 5000 characters. "
        "This is the blueprint the AI uses to plan the entire book — "
        "characters, arcs, world, emotional range, stakes.",
        s["body"],
    ))
    story.append(P("What the AI uses this for", s["h3"]))
    story += bullets([
        "<b>The Storyteller</b> reads it and plans all N chapters upfront — "
        "title, summary, key events, characters involved, and emotional arc "
        "for each.",
        "It builds the full <b>character bible</b> (names, roles, "
        "motivations, arcs) from here. Any character not implied by your "
        "description may be invented by the AI.",
        "The <b>LaunchChapterPlanner</b> turns the plan into a Chapter 1 "
        "conversion brief: first-200-word hook, protagonist sympathy, special "
        "edge tease, early tag proof, progression reward, and cliffhanger.",
        "It seeds the <b>initial running summary</b> that the Writer uses "
        "from chapter 1 onwards.",
        "It drives the <b>continuity ledger</b> — world rules, "
        "significant items, plot seeds — so later chapters stay faithful "
        "to the world you implied.",
    ], s["bullet"])
    story.append(P("How to fill it in well", s["h3"]))
    story.append(P(
        "Think of this as a detailed pitch. Aim for 300–1500 words of real "
        "substance. Structure it any way you like, but cover the areas "
        "below. Anything you leave out is a decision the AI will make for "
        "you.",
        s["body"],
    ))
    story.append(P("What to include", s["h3"]))
    story += bullets([
        "<b>Premise.</b> One or two sentences that state the hook.",
        "<b>Protagonist(s).</b> Name, age, occupation, what they want, "
        "what is in their way, and a meaningful flaw.",
        "<b>Key supporting cast.</b> 2–5 named characters with one line "
        "each on role and relationship to the protagonist.",
        "<b>Setting.</b> Time period, place, atmosphere. Be specific — "
        "<i>1990s Lagos, rainy season, a house on Bourdillon Road</i> "
        "beats <i>modern city</i>.",
        "<b>Central conflict.</b> What is at stake, for whom, and why now.",
        "<b>Tone and genre.</b> Literary thriller? Slow-burn romance? "
        "Folkloric horror? Say it plainly.",
        "<b>Rough arc.</b> Beginning, turning point, ending — even two "
        "sentences each helps enormously.",
        "<b>Hard constraints.</b> Anything the AI must or must not do "
        "(e.g. no magic, stays in one POV, no character dies).",
    ], s["bullet"])
    story.append(P(
        "Every named character, object, and place you mention here becomes "
        "part of the continuity ledger from chapter 1. The AI will not "
        "forget them. If you don't name someone, the AI may invent them — "
        "and then be stuck with that invention.",
        s["callout"],
    ))

    story.append(P("Good example (condensed)", s["h3"]))
    story.append(P(
        '"Set in 1998 Lagos. Ifeoma, a 34-year-old pharmacist running her '
        "late father's shop on Agege Motor Road, finds a ledger hidden "
        "behind a crate of paracetamol showing that her father was "
        "laundering money for a state commissioner. She has two weeks "
        "before an audit. Her younger brother Obi, a junior civil servant, "
        "is named in the ledger and does not yet know. Her mother, "
        "Adaeze, is in early-stage dementia and alternates between lucid "
        "grief and repeating a nursery rhyme from their village. Literary "
        "thriller, slow-burn, first person past tense from Ifeoma's POV. "
        "The tone should be dry and observant — humour used to deflect "
        "from pain. No violence on the page; violence happens off-screen "
        "and lands in conversations. The commissioner, Chief Dike, is the "
        "antagonist but only appears in chapters 4, 8, and 11. Ending: "
        "Ifeoma burns the ledger but keeps a single page — ambiguous "
        'whether she will use it."',
        s["quote"],
    ))
    story.append(P("Weak example (avoid)", s["h3"]))
    story.append(P(
        '"A woman finds a secret from her father. She has to decide what '
        "to do. It is set in Nigeria and has family drama. There is some "
        'suspense."',
        s["warn"],
    ))

    # -------- num_chapters --------
    story.append(P("Field: num_chapters", s["h2"]))
    story.append(P(
        "Integer between 1 and 30. The full length of the book.",
        s["body"],
    ))
    story.append(P("What the AI uses this for", s["h3"]))
    story += bullets([
        "The Storyteller plans <i>exactly</i> this many chapters up front, "
        "pacing the arc across them.",
        "Each chapter is written to a <b>minimum of 2000 words</b> (target "
        "2500–3000), so a 15-chapter book is roughly 30k–45k words.",
        "The AI never writes fewer or more chapters than you ask for.",
    ], s["bullet"])
    story.append(P("How to choose", s["h3"]))
    story += bullets([
        "Short story / novella: 3–6 chapters.",
        "Standard novella: 8–12 chapters.",
        "Full novel within the 30-chapter cap: 15–25 chapters.",
        "If you are unsure, start smaller. You can always start a new book; "
        "you cannot easily cut in half a book that was planned too long.",
    ], s["bullet"])

    # -------- Webnovel preferences --------
    story.append(P("Optional Webnovel preference fields", s["h2"]))
    story.append(P(
        "These fields steer the new Webnovel strategy layer. You can leave "
        "all of them blank and let the AI infer the best market position "
        "from your title and description, but they are useful when you know "
        "the exact reader lane you want.",
        s["body"],
    ))
    story += bullets([
        "<b>platform_genre</b> — preferred Webnovel category, such as "
        "Fantasy, Urban, Games, Eastern, Teen, LGBT+, or General.",
        "<b>lead_type</b> — male lead, female lead, ensemble, or another "
        "reader-facing lane.",
        "<b>target_reader</b> — who should enjoy this serial and what they "
        "already like.",
        "<b>content_rating</b> — general, teen, mature, R18, or your own "
        "house label.",
        "<b>tags</b> — Webnovel-style discovery tags the story must honestly "
        "deliver, such as WEAKTOSTRONG, REVENGE, SYSTEM, BETRAYAL, or ROMANCE.",
        "<b>must_include_tropes</b> — tropes you want built into the plan.",
        "<b>must_avoid_tropes</b> — tropes, content, or patterns the AI must "
        "stay away from.",
        "<b>release_goal</b> — desired launch rhythm, stockpile, or update "
        "target.",
    ], s["bullet"])
    story.append(P(
        "Do not add popular tags just because they are popular. If the prose "
        "does not actually satisfy the tag, readers will feel misled and the "
        "book will attract the wrong audience.",
        s["warn"],
    ))
    story.append(P("Automatic launch chapter plan", s["h2"]))
    story.append(P(
        "After the full story plan is created, the platform now runs a "
        "dedicated LaunchChapterPlanner. You do not fill in extra form fields "
        "for this. It creates an internal brief that the Writer, Reviewer, "
        "Perfectionist, and Publisher all use for Chapter 1.",
        s["body"],
    ))
    story += bullets([
        "<b>First 200 words.</b> The opening must show pressure, desire, "
        "humiliation, danger, mystery, betrayal, romantic tension, or a clear "
        "unfairness before broad setup.",
        "<b>Protagonist snapshot.</b> Readers should quickly know what the "
        "lead wants, what limits them, and why they are worth following.",
        "<b>Special edge tease.</b> The system, rebirth knowledge, rare talent, "
        "hidden status, or repeatable advantage should be hinted early.",
        "<b>Tag proof.</b> The launch batch must visibly satisfy the chosen "
        "tags instead of merely listing them.",
        "<b>Progression reward.</b> Chapter 1 must deliver at least one small "
        "win, clue, leverage shift, power signal, relationship spark, or "
        "survival step.",
        "<b>Cliffhanger.</b> The ending should make Chapter 2 feel necessary, "
        "not optional.",
    ], s["bullet"])
    story.append(P(
        "This is why the description should include the protagonist's desire, "
        "stakes, limitation, and special edge clearly. If those details are "
        "missing, the launch planner has to infer them.",
        s["tip"],
    ))

    # -------- profile_id --------
    story.append(P("Field: profile_id", s["h2"]))
    story.append(P(
        "The ID returned when you generated your profile (Part 1). Paste "
        "it here to make this book sound like you.",
        s["body"],
    ))
    story.append(P("What happens when you provide it", s["h3"]))
    story += bullets([
        "The AI <b>skips Phase 1 entirely</b> — no new Profiler, Empath, "
        "or Masterclass calls.",
        "Your stored <i>author_profile</i>, <i>emotional_guidelines</i>, "
        "and <i>expert_styles</i> are injected directly into the "
        "Storyteller and the Writer.",
        "The book starts faster and sounds closer to you than any "
        "generic run ever could.",
    ], s["bullet"])
    story.append(P("What happens when you omit it", s["h3"]))
    story.append(P(
        "The AI runs Phase 1 with <b>generic prompts</b>, building a "
        "competent but anonymous author voice from the book title and "
        "description alone. Readable, but not yours.",
        s["body"],
    ))
    story.append(P("What happens when the ID is invalid", s["h3"]))
    story.append(P(
        "The request returns <b>400 Bad Request</b>. Nothing is written. "
        "Double-check the ID you saved from Part 1.",
        s["body"],
    ))
    story.append(P(
        "Once you have a profile, always pass profile_id. There is no "
        "reason to run a book without it.",
        s["tip"],
    ))

    # -------- initial_chapters --------
    story.append(P("Field: initial_chapters", s["h2"]))
    story.append(P(
        "How many chapters to draft in this first run. Defaults to 3. "
        "Clamped between 1 and num_chapters.",
        s["body"],
    ))
    story.append(P("Why this exists", s["h3"]))
    story.append(P(
        "Writing a chapter is expensive (it is a real AI generation, not a "
        "template). For long books, you probably want to read the first "
        "few chapters and decide whether the voice feels right <i>before</i> "
        "committing to 25 more. That is exactly what initial_chapters "
        "controls.",
        s["body"],
    ))
    story.append(P("How it works", s["h3"]))
    story += bullets([
        "The Storyteller always plans the <b>full</b> book, regardless of "
        "this value.",
        "The Writer loop only drafts up to <i>initial_chapters</i>.",
        "When it finishes, the story parks in status "
        "<b>awaiting_continue</b>.",
        "You then call <b>POST /api/stories/&lt;id&gt;/continue/</b> with "
        "<code>{\"additional_chapters\": N}</code> to draft the next batch.",
        "If initial_chapters equals num_chapters, the book is written in "
        "one run and finalised immediately.",
    ], s["bullet"])
    story.append(P("Suggested values", s["h3"]))
    story += bullets([
        "<b>3</b> — default. Good for most writers. Read, judge, resume.",
        "<b>1</b> — ultra-careful. Useful for a new profile you have not "
        "tested yet.",
        "<b>= num_chapters</b> — write it all in one shot. Use when you "
        "trust the profile and want the whole manuscript at once.",
    ], s["bullet"])

    # -------- full request --------
    story.append(PageBreak())
    story.append(P("Full request example", s["h2"]))
    story.append(P(
        'POST /api/stories/create/\n'
        'Content-Type: application/json\n\n'
        '{\n'
        '  "book_title": "Things We Do Not Bury",\n'
        '  "book_description": "Set in 1998 Lagos. Ifeoma, a 34-year-old '
        "pharmacist running her late father's shop on Agege Motor Road, "
        'finds a ledger...<continue for 300–1500 words>",\n'
        '  "num_chapters": 15,\n'
        '  "platform_genre": "Urban",\n'
        '  "lead_type": "female lead",\n'
        '  "target_reader": "Readers who like family secrets, revenge, and slow-burn suspense.",\n'
        '  "content_rating": "mature",\n'
        '  "tags": ["REVENGE", "FAMILYSECRET", "STRONGFL"],\n'
        '  "must_include_tropes": ["hidden evidence", "status reversal"],\n'
        '  "must_avoid_tropes": ["random magical rescue"],\n'
        '  "release_goal": "Launch with 5 chapters, then update daily.",\n'
        '  "profile_id": "b4e9-your-saved-profile-id",\n'
        '  "initial_chapters": 3\n'
        '}',
        s["mono"],
    ))
    story.append(P("Response", s["h3"]))
    story.append(P(
        '{\n'
        '  "story_id": "c22a...",\n'
        '  "status": "pending",\n'
        '  "profile_used": true,\n'
        '  "initial_chapters": 3,\n'
        '  "num_chapters": 15,\n'
        '  "webnovel_preferences": {\n'
        '    "platform_genre": "Urban",\n'
        '    "lead_type": "female lead",\n'
        '    "tags": ["REVENGE", "FAMILYSECRET", "STRONGFL"]\n'
        '  }\n'
        '}',
        s["mono"],
    ))
    story.append(P(
        "Generation runs in the background. Poll "
        "<b>GET /api/stories/&lt;story_id&gt;/</b> to watch progress. When "
        "status becomes <i>awaiting_continue</i>, read the draft and, if "
        "you are happy, call the <i>continue</i> endpoint to request more "
        "chapters.",
        s["body"],
    ))

    story.append(PageBreak())
    return story


def checklist_section(s):
    story = []
    story.append(P("Quick checklists", s["h1"]))

    story.append(P("Before you submit a profile", s["h2"]))
    story += bullets([
        "Name is set.",
        "Background names real places, times, or subcultures — no "
        "generalities.",
        "Personality includes at least one thing that annoys you and one "
        "thing you are afraid of.",
        "Communication style describes sentence length and at least one "
        "punctuation habit.",
        "Quirks include words you overuse and patterns you repeat.",
        "At least 2 writing_samples are included, each 300+ words.",
        "Every writing sample is yours. No other writer's work is mixed in.",
    ], s["bullet"])

    story.append(P("Before you submit a story", s["h2"]))
    story += bullets([
        "Title is evocative, not a placeholder.",
        "Description is 300–1500 words of real substance.",
        "Every main character is <b>named</b> in the description.",
        "Setting is specific: place + time + atmosphere.",
        "Central conflict, stakes, and tone are stated plainly.",
        "Any hard constraints (POV, off-screen events, no magic, etc.) are "
        "written down.",
        "Optional Webnovel preference fields only include tags and tropes the "
        "chapters can honestly deliver.",
        "The first chapter promise is clear: hook, protagonist desire, stakes, "
        "special edge, progression reward, and next-chapter pull.",
        "num_chapters is realistic for the amount of story you described.",
        "profile_id is set to your saved profile.",
        "initial_chapters is chosen deliberately (default 3 is fine).",
    ], s["bullet"])

    story.append(P("Common mistakes that ruin output", s["h2"]))
    story += bullets([
        "<b>Vague descriptions.</b> Generic descriptions produce generic "
        "books. The AI cannot invent the specifics you withheld.",
        "<b>Skipping writing_samples.</b> Without samples, your profile is "
        "mostly guesswork based on bio fields.",
        "<b>Forgetting profile_id.</b> Your story will sound fine — but it "
        "will not sound like you.",
        "<b>Over-requesting chapters.</b> Setting num_chapters=30 when the "
        "premise only supports 10 produces padding. Match length to story.",
        "<b>Unnamed characters.</b> If you call someone \"the protagonist's "
        "mother\", the AI will name her for you — and you may not like it.",
        "<b>Contradicting the profile.</b> If your profile says \"literary, "
        "avoid purple prose\" and your description asks for \"lush, "
        "opulent, poetic\" — the AI will compromise, badly.",
    ], s["bullet"])

    return story


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_pdf():
    s = build_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Web Novel Writer's Guide",
        author="AI Writer Platform",
    )

    flowables = []
    flowables += cover(s)
    flowables += profiles_section(s)
    flowables += stories_section(s)
    flowables += checklist_section(s)

    doc.build(flowables)
    print(f"✓ PDF written to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
