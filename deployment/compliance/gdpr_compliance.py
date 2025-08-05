"""GDPR Compliance implementation for Autoformalize.

This module implements GDPR (General Data Protection Regulation) compliance
features including data anonymization, consent management, and audit trails.
"""

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from ..utils.logging_config import setup_logger


class ConsentType(Enum):
    """Types of user consent."""
    PROCESSING = "processing"
    ANALYTICS = "analytics"
    MARKETING = "marketing"
    THIRD_PARTY = "third_party"


class DataCategory(Enum):
    """Categories of personal data."""
    IDENTITY = "identity"
    CONTACT = "contact"  
    USAGE = "usage"
    TECHNICAL = "technical"
    CONTENT = "content"


@dataclass
class ConsentRecord:
    """Record of user consent."""
    user_id: str
    consent_type: ConsentType
    granted: bool
    timestamp: datetime
    version: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "consent_type": self.consent_type.value,
            "granted": self.granted,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent
        }


@dataclass 
class DataProcessingRecord:
    """Record of data processing activity."""
    record_id: str
    user_id: str
    data_category: DataCategory
    purpose: str
    legal_basis: str
    timestamp: datetime
    retention_period: Optional[int] = None  # Days
    processed_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "user_id": self.user_id,
            "data_category": self.data_category.value,
            "purpose": self.purpose,
            "legal_basis": self.legal_basis,
            "timestamp": self.timestamp.isoformat(),
            "retention_period": self.retention_period,
            "processed_by": self.processed_by
        }


class DataAnonymizer:
    """Anonymizes personal data for GDPR compliance."""
    
    @staticmethod
    def hash_pii(data: str, salt: str = "autoformalize_salt") -> str:
        """Hash personally identifiable information."""
        return hashlib.sha256(f"{data}{salt}".encode()).hexdigest()[:16]
    
    @staticmethod
    def anonymize_ip(ip_address: str) -> str:
        """Anonymize IP address by masking last octet."""
        if not ip_address:
            return ""
        
        parts = ip_address.split('.')
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.{parts[2]}.0"
        return "anonymized"
    
    @staticmethod
    def anonymize_latex_content(content: str) -> str:
        """Anonymize LaTeX content by removing potential PII."""
        import re
        
        # Remove email addresses
        content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', content)
        
        # Remove URLs
        content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', content)
        
        # Remove potential names (simple heuristic)
        content = re.sub(r'\\author\{[^}]+\}', r'\\author{[AUTHOR]}', content)
        content = re.sub(r'\\title\{[^}]+\}', r'\\title{[TITLE]}', content)
        
        return content
    
    @classmethod
    def anonymize_request_data(cls, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize request data for storage."""
        anonymized = request_data.copy()
        
        # Anonymize LaTeX content
        if 'latex_content' in anonymized:
            anonymized['latex_content'] = cls.anonymize_latex_content(anonymized['latex_content'])
        
        # Remove or hash user identifiers
        if 'user_id' in anonymized:
            anonymized['user_id'] = cls.hash_pii(anonymized['user_id'])
        
        if 'ip_address' in anonymized:
            anonymized['ip_address'] = cls.anonymize_ip(anonymized['ip_address'])
        
        # Remove user agent details
        if 'user_agent' in anonymized:
            anonymized['user_agent'] = '[USER_AGENT]'
        
        return anonymized


class ConsentManager:
    """Manages user consent for GDPR compliance."""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.consent_records: Dict[str, List[ConsentRecord]] = {}
        self.consent_version = "1.0"
    
    async def record_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        granted: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> ConsentRecord:
        """Record user consent."""
        
        # Anonymize IP address
        anonymized_ip = DataAnonymizer.anonymize_ip(ip_address) if ip_address else None
        
        consent = ConsentRecord(
            user_id=user_id,
            consent_type=consent_type,
            granted=granted,
            timestamp=datetime.utcnow(),
            version=self.consent_version,
            ip_address=anonymized_ip,
            user_agent="[USER_AGENT]"  # Anonymized
        )
        
        if user_id not in self.consent_records:
            self.consent_records[user_id] = []
        
        self.consent_records[user_id].append(consent)
        
        self.logger.info(f"Recorded consent: user={user_id}, type={consent_type.value}, granted={granted}")
        
        return consent
    
    async def check_consent(self, user_id: str, consent_type: ConsentType) -> bool:
        """Check if user has granted specific consent."""
        if user_id not in self.consent_records:
            return False
        
        # Get most recent consent for this type
        user_consents = self.consent_records[user_id]
        relevant_consents = [c for c in user_consents if c.consent_type == consent_type]
        
        if not relevant_consents:
            return False
        
        # Return most recent consent status
        latest_consent = max(relevant_consents, key=lambda x: x.timestamp)
        return latest_consent.granted
    
    async def withdraw_consent(self, user_id: str, consent_type: ConsentType) -> bool:
        """Withdraw user consent."""
        return await self.record_consent(user_id, consent_type, granted=False)
    
    async def get_consent_history(self, user_id: str) -> List[ConsentRecord]:
        """Get consent history for user."""
        return self.consent_records.get(user_id, [])
    
    async def export_user_consents(self, user_id: str) -> Dict[str, Any]:
        """Export user consent data (for data portability)."""
        consents = await self.get_consent_history(user_id)
        
        return {
            "user_id": user_id,
            "consents": [consent.to_dict() for consent in consents],
            "exported_at": datetime.utcnow().isoformat()
        }


class DataProcessor:
    """Manages data processing for GDPR compliance."""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.processing_records: List[DataProcessingRecord] = []
        self.retention_policies: Dict[DataCategory, int] = {
            DataCategory.IDENTITY: 30,      # 30 days
            DataCategory.CONTACT: 365,      # 1 year
            DataCategory.USAGE: 730,        # 2 years
            DataCategory.TECHNICAL: 90,     # 90 days
            DataCategory.CONTENT: 1095,     # 3 years
        }
    
    async def record_processing(
        self,
        user_id: str,
        data_category: DataCategory,
        purpose: str,
        legal_basis: str,
        processed_by: Optional[str] = None
    ) -> DataProcessingRecord:
        """Record data processing activity."""
        
        record = DataProcessingRecord(
            record_id=str(uuid.uuid4()),
            user_id=user_id,
            data_category=data_category,
            purpose=purpose,
            legal_basis=legal_basis,
            timestamp=datetime.utcnow(),
            retention_period=self.retention_policies.get(data_category),
            processed_by=processed_by
        )
        
        self.processing_records.append(record)
        
        self.logger.info(f"Recorded processing: user={user_id}, category={data_category.value}, purpose={purpose}")
        
        return record
    
    async def get_user_processing_records(self, user_id: str) -> List[DataProcessingRecord]:
        """Get all processing records for a user."""
        return [record for record in self.processing_records if record.user_id == user_id]
    
    async def delete_expired_data(self) -> int:
        """Delete data that has exceeded retention period."""
        now = datetime.utcnow()
        deleted_count = 0
        
        # Check each processing record
        for record in self.processing_records[:]:  # Copy to avoid modification during iteration
            if record.retention_period:
                expiry_date = record.timestamp + timedelta(days=record.retention_period)
                if now > expiry_date:
                    # In real implementation, would delete actual data
                    self.processing_records.remove(record)
                    deleted_count += 1
                    self.logger.info(f"Deleted expired data record: {record.record_id}")
        
        return deleted_count
    
    async def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all data for a user (data portability)."""
        records = await self.get_user_processing_records(user_id)
        
        return {
            "user_id": user_id,
            "processing_records": [record.to_dict() for record in records],
            "exported_at": datetime.utcnow().isoformat(),
            "retention_policies": {cat.value: days for cat, days in self.retention_policies.items()}
        }


class GDPRCompliance:
    """Main GDPR compliance coordinator."""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.consent_manager = ConsentManager()
        self.data_processor = DataProcessor()
        self.anonymizer = DataAnonymizer()
        
        # Required consent types for different operations
        self.required_consents = {
            "formalization": [ConsentType.PROCESSING],
            "analytics": [ConsentType.PROCESSING, ConsentType.ANALYTICS],
            "api_usage": [ConsentType.PROCESSING],
        }
    
    async def check_processing_lawful(
        self,
        user_id: str,
        operation: str,
        data_category: DataCategory
    ) -> bool:
        """Check if data processing is lawful under GDPR."""
        
        # Check required consents
        required_consents = self.required_consents.get(operation, [])
        
        for consent_type in required_consents:
            if not await self.consent_manager.check_consent(user_id, consent_type):
                self.logger.warning(f"Missing consent: user={user_id}, type={consent_type.value}")
                return False
        
        return True
    
    async def process_formalization_request(
        self,
        user_id: str,
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process formalization request with GDPR compliance."""
        
        # Check consent
        if not await self.check_processing_lawful(user_id, "formalization", DataCategory.CONTENT):
            raise ValueError("User consent required for processing")
        
        # Record processing activity
        await self.data_processor.record_processing(
            user_id=user_id,
            data_category=DataCategory.CONTENT,
            purpose="Mathematical formalization",
            legal_basis="Consent (Art. 6(1)(a))",
            processed_by="autoformalize-system"
        )
        
        # Anonymize request data for storage
        anonymized_data = self.anonymizer.anonymize_request_data(request_data)
        
        return anonymized_data
    
    async def handle_data_subject_request(
        self,
        user_id: str,
        request_type: str
    ) -> Dict[str, Any]:
        """Handle data subject rights requests."""
        
        if request_type == "access":
            # Right of access (Art. 15)
            consent_data = await self.consent_manager.export_user_consents(user_id)
            processing_data = await self.data_processor.export_user_data(user_id)
            
            return {
                "request_type": "access",
                "user_id": user_id,
                "data": {
                    "consents": consent_data,
                    "processing": processing_data
                },
                "generated_at": datetime.utcnow().isoformat()
            }
        
        elif request_type == "portability":
            # Right to data portability (Art. 20)
            return await self.data_processor.export_user_data(user_id)
        
        elif request_type == "erasure":
            # Right to erasure (Art. 17)
            deleted_consents = len(await self.consent_manager.get_consent_history(user_id))
            deleted_processing = len(await self.data_processor.get_user_processing_records(user_id))
            
            # In real implementation, would delete actual data
            if user_id in self.consent_manager.consent_records:
                del self.consent_manager.consent_records[user_id]
            
            self.data_processor.processing_records = [
                record for record in self.data_processor.processing_records
                if record.user_id != user_id
            ]
            
            return {
                "request_type": "erasure",
                "user_id": user_id,
                "deleted_records": {
                    "consents": deleted_consents,
                    "processing": deleted_processing
                },
                "processed_at": datetime.utcnow().isoformat()
            }
        
        elif request_type == "rectification":
            # Right to rectification (Art. 16)
            return {
                "request_type": "rectification",
                "user_id": user_id,
                "message": "Please contact support to update your data",
                "processed_at": datetime.utcnow().isoformat()
            }
        
        else:
            raise ValueError(f"Unknown request type: {request_type}")
    
    async def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate GDPR compliance report."""
        
        # Count consents by type
        consent_stats = {}
        for user_consents in self.consent_manager.consent_records.values():
            for consent in user_consents:
                consent_type = consent.consent_type.value
                if consent_type not in consent_stats:
                    consent_stats[consent_type] = {"granted": 0, "denied": 0}
                
                if consent.granted:
                    consent_stats[consent_type]["granted"] += 1
                else:
                    consent_stats[consent_type]["denied"] += 1
        
        # Count processing by category
        processing_stats = {}
        for record in self.data_processor.processing_records:
            category = record.data_category.value
            processing_stats[category] = processing_stats.get(category, 0) + 1
        
        # Data retention status
        now = datetime.utcnow()
        retention_status = {"compliant": 0, "near_expiry": 0, "overdue": 0}
        
        for record in self.data_processor.processing_records:
            if record.retention_period:
                expiry_date = record.timestamp + timedelta(days=record.retention_period)
                days_to_expiry = (expiry_date - now).days
                
                if days_to_expiry < 0:
                    retention_status["overdue"] += 1
                elif days_to_expiry < 30:
                    retention_status["near_expiry"] += 1
                else:
                    retention_status["compliant"] += 1
        
        return {
            "report_generated": datetime.utcnow().isoformat(),
            "consent_statistics": consent_stats,
            "processing_statistics": processing_stats,
            "retention_compliance": retention_status,
            "total_users": len(self.consent_manager.consent_records),
            "total_processing_records": len(self.data_processor.processing_records),
            "anonymization_enabled": True,
            "retention_policies": {
                cat.value: days for cat, days in self.data_processor.retention_policies.items()
            }
        }
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """Run GDPR maintenance tasks."""
        
        # Delete expired data
        deleted_count = await self.data_processor.delete_expired_data()
        
        # Generate compliance report
        compliance_report = await self.generate_compliance_report()
        
        return {
            "maintenance_run": datetime.utcnow().isoformat(),
            "expired_data_deleted": deleted_count,
            "compliance_status": compliance_report
        }


# Global instance
gdpr_compliance = GDPRCompliance()


async def example_usage():
    """Example usage of GDPR compliance system."""
    
    # Record user consent
    user_id = "user123"
    await gdpr_compliance.consent_manager.record_consent(
        user_id=user_id,
        consent_type=ConsentType.PROCESSING,
        granted=True,
        ip_address="192.168.1.100"
    )
    
    # Process a formalization request
    request_data = {
        "latex_content": "\\theorem{Test}",
        "target_system": "lean4",
        "user_id": user_id
    }
    
    anonymized_data = await gdpr_compliance.process_formalization_request(
        user_id=user_id,
        request_data=request_data
    )
    
    print("Anonymized data:", anonymized_data)
    
    # Handle data subject access request
    access_data = await gdpr_compliance.handle_data_subject_request(
        user_id=user_id,
        request_type="access"
    )
    
    print("Access request data:", json.dumps(access_data, indent=2))
    
    # Generate compliance report
    report = await gdpr_compliance.generate_compliance_report()
    print("Compliance report:", json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(example_usage())