from resumesort.parser import (
    extract_cgpa,
    extract_email,
    extract_github_url,
    extract_years_experience,
    parse_resume_text,
)


def test_extract_basic_fields():
    text = """
    Tejas Venugopalan
    tejas@example.com
    https://github.com/tejasvenu45
    CGPA: 8.74 / 10
    Skills: Python, FastAPI, MongoDB, Docker
    Projects
    Real-Time Warehouse Inventory Management using FastAPI and MongoDB
    President of Coding Club
    """
    profile = parse_resume_text(text, source_name="resume.pdf")

    assert extract_email(text) == "tejas@example.com"
    assert extract_github_url(text) == "https://github.com/tejasvenu45"
    assert extract_cgpa(text) == 8.74
    assert profile.name == "Tejas Venugopalan"
    assert "Python" in profile.skills
    assert profile.leadership is True
    assert profile.projects


def test_extract_years_experience():
    assert extract_years_experience("Software engineer with 5+ years of experience in Python") == 5.0
    assert extract_years_experience("over 3.5 years experience; previously 2 yrs experience") == 3.5
    assert extract_years_experience("No experience numbers here") is None


def test_extract_github_url_without_protocol():
    assert extract_github_url("Profile: github.com/PP-695 and more") == "https://github.com/PP-695"
    assert extract_github_url("no links here") is None


def test_pdf_hyperlink_annotations_are_extracted():
    """Resumes often show 'Github' as link text with the URL only in the annotation."""
    from io import BytesIO

    from fpdf import FPDF

    from resumesort.parser import extract_text_from_pdf, parse_resume_text

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, "PURANDAR BALASA", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(20, 8, "Github", link="https://github.com/PP-695")

    text = extract_text_from_pdf(BytesIO(bytes(pdf.output())))
    profile = parse_resume_text(text)

    assert profile.github_url == "https://github.com/PP-695"
