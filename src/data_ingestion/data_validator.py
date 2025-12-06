"""
Data Validator
==================
Use After:Data_loader.py

Validate data quality before preprocessing to ensure data consistance
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from abc import abstractmethod,ABC

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Single validation issue."""
    check_name: str
    severity: ValidationSeverity
    message: str
    affected_rows: int = 0
    details: Dict = field(default_factory=dict)


@dataclass 
class ValidationReport:
    """Complete validation report."""
    is_valid: bool
    issues: List[ValidationIssue]
    total_rows: int
    valid_rows: int

class BaseValidationRule(ABC):
    @abstractmethod
    def run(self,df:pd.DataFrame) -> Optional[ValidationIssue]:
        pass

class RequiredColumnRule(BaseValidationRule):
    #Essential Columns for training
    REQUIRED = ['order_id', 'customer_id', 'order_date', 'product_name', 'product_variant']

    def run(self,df:pd.DataFrame) -> Optional[ValidationIssue]:
        missing = [c for c in self.REQUIRED if c not in df.columns]
        if missing:
            return ValidationIssue(
                check_name="Required Columns",
                severity= ValidationSeverity.CRITICAL,
                message={f"Missing columns {missing}"},
                affected_rows=len(df), # all rows as missing at column level
                details ={"Missing Columns":missing})
        return None
#Check at critical column level
class NullCheckRule(BaseValidationRule):
        CRITICAL_COLS = ['order_id', 'customer_id', 'order_date', 'product_name']
        
        def run(self,df:pd.DataFrame) -> Optional[ValidationIssue]:

            #Count null in critical cols
            nulls = {c:df[c].isna().sum() for c in self.CRITICAL_COLS if c in df.columns}
            #Keep only columns with >0 null
            violations = {c: n for c, n in nulls.items() if n > 0}
            if violations:
                total = sum(violations.values())
                return ValidationIssue(
                check_name="Null Critical Columns",
                severity= ValidationSeverity.ERROR,
                message={f"Null found : {violations}"},
                affected_rows=total, # all rows as missing at column level
                details =violations)
            return None

class ValidationFactory:
    @staticmethod
    def default_rules() -> List[BaseValidationRule]:
        return [
            RequiredColumnRule(),
            NullCheckRule()
            #Add into list if more rule need to be added
        ]

class DataValidator:
    def __init__(self, rules: Optional[List[BaseValidationRule]] = None, strict_mode=False):
        self.rules = rules or ValidationFactory.default_rules()
        self.strict_mode = strict_mode
        """
        Using strcit mode bool to define how strict the validation engine should be
        If True = fail at warning level
        If False - fail at ERROR level
        """

    def validate(self, df:pd.DataFrame) -> ValidationReport:
        issues = []

        for rule in self.rules:
            issue = rule.run(df)
            if issue:
                issues.append(issue)
        is_valid = True
        for issue in issues:
            if self.strict_mode == True and issue.severity in [ValidationSeverity.WARNING, ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                is_valid = False
            elif self.strict_mode == False and issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                is_valid = False        
        valid_rows = len(df) - sum(
            issue.affected_rows for issue in issues if issue.severity == ValidationSeverity.CRITICAL
        )

        return ValidationReport(
            is_valid=is_valid,
            issues=issues,
            total_rows=len(df),
            valid_rows=valid_rows
        )
    
if __name__ == "__main__":
    pass
"""
Use example
#Validate df
loader = DataLoader(CSVIngestion("path/to/file"))
raw = loader.load()

#Load via bigquery
loader= DataLoader(BigQueryIngestion(project_id="my-gcp-id",
    query="SELECT * FROM dataset.transactions"))
"""