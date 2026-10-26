"""Structured schemas for rental agreement metadata.

Two layers:

``AgreementExtraction``
    What the language model fills in. It keeps easily-confused concepts apart
    (execution date vs tenancy start, monthly rent vs deposit vs advance,
    termination vs renewal notice, company vs its representative) and carries a
    verbatim evidence quote for every value. Its JSON schema is passed to
    Ollama's ``format`` so generation is constrained to this shape.

``AgreementMetadata``
    The six assignment fields, serialised with the dataset's exact CSV column
    names (including the "Aggrement" spelling). ``from_extraction`` maps one to
    the other; the mapping follows the label definitions in docs/schema.md.

Missing values are ``None`` (JSON ``null``, empty CSV cell); validators reject
malformed values, such as impossible calendar dates, instead of guessing.

The evaluation-only normalisation lives separately in ``src/normalize.py``.
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from src.config import LABEL_COLUMNS

DATE_FORMAT = "%d.%m.%Y"  # DD.MM.YYYY, as in the reference CSVs
DAYS_PER_UNIT = {"days": 1, "weeks": 7, "months": 30}  # label convention: 1 month = 30 days

EVIDENCE = "Exact quote from the document supporting this value, or null if the value is null."


def _blank_to_none(value):
    """Collapse whitespace; an empty or whitespace-only string means "missing"."""
    if isinstance(value, str):
        value = " ".join(value.split())
        return value or None
    return value


def _valid_date(value: str | None) -> str | None:
    """Accept only real calendar dates and re-emit them zero-padded (1.4.2010 -> 01.04.2010)."""
    if value is None:
        return None
    try:
        return datetime.strptime(value, DATE_FORMAT).strftime(DATE_FORMAT)
    except ValueError as exc:
        raise ValueError(f"'{value}' is not a valid DD.MM.YYYY calendar date") from exc


Text = Annotated[str | None, BeforeValidator(_blank_to_none)]
Date = Annotated[str | None, BeforeValidator(_blank_to_none), AfterValidator(_valid_date)]


class _Strict(BaseModel):
    """Reject unknown keys so the generated JSON schema is closed."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- LLM output


class Party(_Strict):
    name: Text = Field(
        description="Legal name of the contracting party exactly as written (a company name if the "
        "party is a company). Exclude addresses, ages and relations such as S/o."
    )
    representative: Text = Field(
        description="Person signing or acting for the party (e.g. a company director or power-of-attorney "
        "holder), if different from the party itself; otherwise null."
    )
    evidence: Text = Field(description=EVIDENCE)


class Amount(_Strict):
    value: int | None = Field(ge=0, description="Whole number, no currency symbol, separators or decimals.")
    evidence: Text = Field(description=EVIDENCE)


class DateValue(_Strict):
    value: Date = Field(description="Date as DD.MM.YYYY, or null if not stated.")
    evidence: Text = Field(description=EVIDENCE)


class NoticePeriod(_Strict):
    value: int | None = Field(gt=0, description="Number as stated, e.g. 2 for 'two months'.")
    unit: Literal["days", "weeks", "months"] | None = Field(description="Unit as stated in the document.")
    evidence: Text = Field(description=EVIDENCE)

    @model_validator(mode="after")
    def _value_and_unit_together(self):
        if (self.value is None) != (self.unit is None):
            raise ValueError("notice value and unit must both be given or both be null")
        return self

    @property
    def days(self) -> int | None:
        return None if self.value is None else self.value * DAYS_PER_UNIT[self.unit]


class AgreementExtraction(_Strict):
    """Everything the model reads from one agreement, with evidence."""

    party_one: Party = Field(description="Lessor / landlord / owner: the party letting the property.")
    party_two: Party = Field(description="Lessee / tenant: the party renting the property.")
    monthly_rent: Amount = Field(description="Recurring rent per month. Not the deposit or advance.")
    security_deposit: Amount = Field(description="Refundable security deposit, if any.")
    advance_rent: Amount = Field(description="Rent paid in advance (not a deposit), if any.")
    execution_date: DateValue = Field(description="Date the agreement was signed / executed / entered into.")
    tenancy_start_date: DateValue = Field(
        description="Date the tenancy or lease period begins (commencement), if stated."
    )
    tenancy_end_date: DateValue = Field(
        description="Date the tenancy ends: stated directly, or the last day of the stated term "
        "counted from the start date."
    )
    term_months: int | None = Field(gt=0, description="Length of the tenancy in months, if stated.")
    termination_notice: NoticePeriod = Field(
        description="Notice either party must give to terminate the agreement or vacate the premises."
    )
    renewal_notice: NoticePeriod = Field(
        description="Notice required to renew or extend the agreement, only if a separate renewal "
        "notice clause exists."
    )

    def submission_sources(self) -> dict[str, "Party | Amount | DateValue | NoticePeriod"]:
        """Which extracted field feeds each submission column (definitions: docs/schema.md)."""
        start = self.tenancy_start_date if self.tenancy_start_date.value else self.execution_date
        notice = self.termination_notice if self.termination_notice.value is not None else self.renewal_notice
        return dict(zip(LABEL_COLUMNS, [self.monthly_rent, start, self.tenancy_end_date, notice,
                                        self.party_one, self.party_two]))


# --------------------------------------------------------------------------- submission record


class AgreementMetadata(BaseModel):
    """The six assignment fields with the dataset's CSV column names."""

    model_config = ConfigDict(populate_by_name=True)

    agreement_value: int | None = Field(default=None, ge=0, alias=LABEL_COLUMNS[0])
    agreement_start_date: Date = Field(default=None, alias=LABEL_COLUMNS[1])
    agreement_end_date: Date = Field(default=None, alias=LABEL_COLUMNS[2])
    renewal_notice_days: int | None = Field(default=None, gt=0, alias=LABEL_COLUMNS[3])
    party_one: Text = Field(default=None, alias=LABEL_COLUMNS[4])
    party_two: Text = Field(default=None, alias=LABEL_COLUMNS[5])

    @classmethod
    def from_extraction(cls, x: AgreementExtraction) -> "AgreementMetadata":
        """Map the detailed extraction to the six fields (definitions: docs/schema.md)."""
        rent, start, end, notice, one, two = x.submission_sources().values()
        return cls(
            agreement_value=rent.value,
            agreement_start_date=start.value,
            agreement_end_date=end.value,
            renewal_notice_days=notice.days,
            party_one=one.name,
            party_two=two.name,
        )

    def to_json(self) -> str:
        """JSON with CSV column names; missing values are null."""
        return self.model_dump_json(by_alias=True)

    def to_csv_row(self) -> dict[str, str]:
        """CSV cells keyed by column name; missing values are empty strings."""
        return {k: "" if v is None else str(v) for k, v in self.model_dump(by_alias=True).items()}
