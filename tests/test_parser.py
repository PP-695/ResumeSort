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
