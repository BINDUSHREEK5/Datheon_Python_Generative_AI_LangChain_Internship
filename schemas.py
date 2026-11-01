
"""Pydantic schemas for invoice and resume extraction."""

import re
from datetime import date
from decimal import Decimal

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    """Reject unexpected fields and require explicitly declared keys."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- Invoice ---


class LineItem(StrictModel):
    description: str | None
    quantity: Decimal | None = Field(ge=0)
    amount: Decimal | None


class Invoice(StrictModel):
    invoice_number: str | None
    vendor: str | None
    invoice_date: date | None
    due_date: date | None
    total_amount: Decimal | None = Field(ge=0)
    currency: str | None
    line_items: list[LineItem]

    @field_validator("currency")
    @classmethod
    def currency_must_be_iso_code(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Z]{3}", value):
            raise ValueError(
                "currency must be a 3-letter uppercase code such as 'INR' or 'USD'"
            )
        return value

    @model_validator(mode="after")
    def due_date_not_before_invoice_date(self) -> "Invoice":
        if self.invoice_date and self.due_date and self.due_date < self.invoice_date:
            raise ValueError("due_date cannot be earlier than invoice_date")
        return self


# ----------------------------------------------------------------- Resume ---


class Education(StrictModel):
    institution: str | None
    degree: str | None
    field_of_study: str | None
    start_date: str | None
    end_date: str | None


class WorkExperience(StrictModel):
    company: str | None
    job_title: str | None
    start_date: str | None
    end_date: str | None
    responsibilities: list[str]


class Project(StrictModel):
    name: str | None
    description: str | None
    technologies: list[str]


class Resume(StrictModel):
    candidate_name: str | None
    email: EmailStr | None
    phone: str | None
    skills: list[str]
    education: list[Education]
    work_experience: list[WorkExperience]
    projects: list[Project]
