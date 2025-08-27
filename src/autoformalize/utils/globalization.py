"""Globalization and internationalization utilities.

This module provides comprehensive globalization features including:
- Multi-language support (i18n)
- Multi-region deployment capabilities
- Compliance with international regulations
- Cross-platform compatibility
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import locale
from datetime import datetime, timezone

from .logging_config import setup_logger


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"
    RUSSIAN = "ru"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    KOREAN = "ko"


class ComplianceRegion(Enum):
    """Compliance regions with specific regulatory requirements."""
    GDPR = "gdpr"        # European Union
    CCPA = "ccpa"        # California
    PDPA = "pdpa"        # Singapore/Thailand
    PIPEDA = "pipeda"    # Canada
    LGPD = "lgpd"        # Brazil


@dataclass
class LocalizationConfig:
    """Configuration for localization settings."""
    language: SupportedLanguage = SupportedLanguage.ENGLISH
    region: Optional[str] = None
    timezone: str = "UTC"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: str = "en_US"
    currency: str = "USD"
    compliance_regions: List[ComplianceRegion] = field(default_factory=list)


@dataclass 
class MultiRegionConfig:
    """Configuration for multi-region deployment."""
    primary_region: str = "us-east-1"
    secondary_regions: List[str] = field(default_factory=lambda: ["eu-west-1", "ap-southeast-1"])
    data_residency_requirements: Dict[str, List[str]] = field(default_factory=dict)
    compliance_by_region: Dict[str, List[ComplianceRegion]] = field(default_factory=dict)
    cdn_endpoints: Dict[str, str] = field(default_factory=dict)


class InternationalizationManager:
    """Manager for internationalization and localization."""
    
    def __init__(self, localization_config: Optional[LocalizationConfig] = None):
        self.config = localization_config or LocalizationConfig()
        self.logger = setup_logger(__name__)
        self.translations: Dict[str, Dict[str, str]] = {}
        self._load_translations()
        
        # Set system locale if possible
        self._set_system_locale()
    
    def _set_system_locale(self) -> None:
        """Set system locale based on configuration."""
        try:
            if self.config.language == SupportedLanguage.ENGLISH:
                locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
            elif self.config.language == SupportedLanguage.SPANISH:
                locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
            elif self.config.language == SupportedLanguage.FRENCH:
                locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')
            elif self.config.language == SupportedLanguage.GERMAN:
                locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
            elif self.config.language == SupportedLanguage.JAPANESE:
                locale.setlocale(locale.LC_ALL, 'ja_JP.UTF-8')
            elif self.config.language == SupportedLanguage.CHINESE:
                locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
        except locale.Error as e:
            self.logger.warning(f"Could not set locale: {e}. Using system default.")
    
    def _load_translations(self) -> None:
        """Load translation files for supported languages."""
        translations_dir = Path(__file__).parent.parent / "translations"
        
        # Create built-in translations if directory doesn't exist
        if not translations_dir.exists():
            self._create_builtin_translations()
        else:
            self._load_translation_files(translations_dir)
    
    def _create_builtin_translations(self) -> None:
        """Create built-in translation dictionaries."""
        
        # English (base language)
        self.translations['en'] = {
            'formalization_started': 'Formalization started',
            'formalization_completed': 'Formalization completed successfully',
            'formalization_failed': 'Formalization failed',
            'parsing_started': 'Starting LaTeX parsing',
            'parsing_completed': 'LaTeX parsing completed',
            'validation_failed': 'Input validation failed',
            'cache_hit': 'Cache hit for formalization request',
            'cache_miss': 'Cache miss for formalization request',
            'health_check_passed': 'Health check passed',
            'health_check_failed': 'Health check failed',
            'processing_time': 'Processing time',
            'success_rate': 'Success rate',
            'error_occurred': 'An error occurred',
            'timeout_exceeded': 'Operation timeout exceeded',
            'invalid_input': 'Invalid input provided',
            'system_ready': 'System is ready',
            'shutdown_initiated': 'Graceful shutdown initiated',
            'configuration_loaded': 'Configuration loaded successfully'
        }
        
        # Spanish
        self.translations['es'] = {
            'formalization_started': 'Formalización iniciada',
            'formalization_completed': 'Formalización completada exitosamente',
            'formalization_failed': 'La formalización falló',
            'parsing_started': 'Iniciando análisis de LaTeX',
            'parsing_completed': 'Análisis de LaTeX completado',
            'validation_failed': 'La validación de entrada falló',
            'cache_hit': 'Acierto de caché para solicitud de formalización',
            'cache_miss': 'Fallo de caché para solicitud de formalización',
            'health_check_passed': 'Verificación de salud pasada',
            'health_check_failed': 'Verificación de salud falló',
            'processing_time': 'Tiempo de procesamiento',
            'success_rate': 'Tasa de éxito',
            'error_occurred': 'Ocurrió un error',
            'timeout_exceeded': 'Tiempo límite de operación excedido',
            'invalid_input': 'Entrada inválida proporcionada',
            'system_ready': 'El sistema está listo',
            'shutdown_initiated': 'Apagado elegante iniciado',
            'configuration_loaded': 'Configuración cargada exitosamente'
        }
        
        # French
        self.translations['fr'] = {
            'formalization_started': 'Formalisation commencée',
            'formalization_completed': 'Formalisation terminée avec succès',
            'formalization_failed': 'La formalisation a échoué',
            'parsing_started': 'Début de l\'analyse LaTeX',
            'parsing_completed': 'Analyse LaTeX terminée',
            'validation_failed': 'La validation d\'entrée a échoué',
            'cache_hit': 'Cache trouvé pour la demande de formalisation',
            'cache_miss': 'Cache manqué pour la demande de formalisation',
            'health_check_passed': 'Vérification de santé réussie',
            'health_check_failed': 'Vérification de santé échouée',
            'processing_time': 'Temps de traitement',
            'success_rate': 'Taux de réussite',
            'error_occurred': 'Une erreur s\'est produite',
            'timeout_exceeded': 'Délai d\'opération dépassé',
            'invalid_input': 'Entrée invalide fournie',
            'system_ready': 'Le système est prêt',
            'shutdown_initiated': 'Arrêt gracieux initié',
            'configuration_loaded': 'Configuration chargée avec succès'
        }
        
        # German
        self.translations['de'] = {
            'formalization_started': 'Formalisierung gestartet',
            'formalization_completed': 'Formalisierung erfolgreich abgeschlossen',
            'formalization_failed': 'Formalisierung fehlgeschlagen',
            'parsing_started': 'LaTeX-Parsing gestartet',
            'parsing_completed': 'LaTeX-Parsing abgeschlossen',
            'validation_failed': 'Eingabevalidierung fehlgeschlagen',
            'cache_hit': 'Cache-Treffer für Formalisierungsanfrage',
            'cache_miss': 'Cache-Verfehlung für Formalisierungsanfrage',
            'health_check_passed': 'Gesundheitsprüfung bestanden',
            'health_check_failed': 'Gesundheitsprüfung fehlgeschlagen',
            'processing_time': 'Verarbeitungszeit',
            'success_rate': 'Erfolgsrate',
            'error_occurred': 'Ein Fehler ist aufgetreten',
            'timeout_exceeded': 'Betriebstimeout überschritten',
            'invalid_input': 'Ungültige Eingabe bereitgestellt',
            'system_ready': 'System ist bereit',
            'shutdown_initiated': 'Ordnungsgemäßes Herunterfahren eingeleitet',
            'configuration_loaded': 'Konfiguration erfolgreich geladen'
        }
        
        # Japanese
        self.translations['ja'] = {
            'formalization_started': '形式化が開始されました',
            'formalization_completed': '形式化が正常に完了しました',
            'formalization_failed': '形式化に失敗しました',
            'parsing_started': 'LaTeX解析を開始しています',
            'parsing_completed': 'LaTeX解析が完了しました',
            'validation_failed': '入力検証に失敗しました',
            'cache_hit': '形式化要求のキャッシュヒット',
            'cache_miss': '形式化要求のキャッシュミス',
            'health_check_passed': 'ヘルスチェックに合格しました',
            'health_check_failed': 'ヘルスチェックに失敗しました',
            'processing_time': '処理時間',
            'success_rate': '成功率',
            'error_occurred': 'エラーが発生しました',
            'timeout_exceeded': '操作タイムアウトを超過しました',
            'invalid_input': '無効な入力が提供されました',
            'system_ready': 'システムは準備完了です',
            'shutdown_initiated': '正常なシャットダウンが開始されました',
            'configuration_loaded': '設定が正常に読み込まれました'
        }
        
        # Chinese (Simplified)
        self.translations['zh'] = {
            'formalization_started': '形式化已开始',
            'formalization_completed': '形式化成功完成',
            'formalization_failed': '形式化失败',
            'parsing_started': '开始LaTeX解析',
            'parsing_completed': 'LaTeX解析完成',
            'validation_failed': '输入验证失败',
            'cache_hit': '形式化请求的缓存命中',
            'cache_miss': '形式化请求的缓存未命中',
            'health_check_passed': '健康检查通过',
            'health_check_failed': '健康检查失败',
            'processing_time': '处理时间',
            'success_rate': '成功率',
            'error_occurred': '发生错误',
            'timeout_exceeded': '操作超时',
            'invalid_input': '提供了无效输入',
            'system_ready': '系统已就绪',
            'shutdown_initiated': '正常关闭已启动',
            'configuration_loaded': '配置加载成功'
        }
    
    def _load_translation_files(self, translations_dir: Path) -> None:
        """Load translation files from directory."""
        for lang_file in translations_dir.glob("*.json"):
            lang_code = lang_file.stem
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    self.translations[lang_code] = json.load(f)
                self.logger.debug(f"Loaded translations for {lang_code}")
            except Exception as e:
                self.logger.warning(f"Failed to load translations for {lang_code}: {e}")
    
    def translate(self, key: str, language: Optional[SupportedLanguage] = None) -> str:
        """Translate a message key to the specified language.
        
        Args:
            key: Translation key
            language: Target language (uses config default if None)
            
        Returns:
            Translated string or original key if translation not found
        """
        lang_code = (language or self.config.language).value
        
        if lang_code in self.translations and key in self.translations[lang_code]:
            return self.translations[lang_code][key]
        
        # Fallback to English
        if 'en' in self.translations and key in self.translations['en']:
            return self.translations['en'][key]
        
        # Return key if no translation found
        return key
    
    def format_datetime(self, dt: datetime) -> str:
        """Format datetime according to locale settings."""
        if self.config.timezone != "UTC":
            # Convert to local timezone (simplified)
            pass
        
        return dt.strftime(f"{self.config.date_format} {self.config.time_format}")
    
    def format_number(self, number: float, decimal_places: int = 2) -> str:
        """Format number according to locale settings."""
        try:
            return f"{number:,.{decimal_places}f}"
        except Exception:
            return str(number)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return [lang.value for lang in SupportedLanguage]


class ComplianceManager:
    """Manager for regulatory compliance across regions."""
    
    def __init__(self, multi_region_config: Optional[MultiRegionConfig] = None):
        self.config = multi_region_config or MultiRegionConfig()
        self.logger = setup_logger(__name__)
        
        # Initialize compliance settings
        self._compliance_settings = self._initialize_compliance_settings()
    
    def _initialize_compliance_settings(self) -> Dict[ComplianceRegion, Dict[str, Any]]:
        """Initialize compliance settings for different regions."""
        return {
            ComplianceRegion.GDPR: {
                'data_retention_days': 365 * 3,  # 3 years
                'anonymization_required': True,
                'consent_required': True,
                'right_to_erasure': True,
                'data_portability': True,
                'privacy_by_design': True,
                'dpo_required': True,  # Data Protection Officer
                'allowed_transfers': ['adequacy_decision', 'bcr', 'scc'],  # Standard Contractual Clauses
                'prohibited_data_types': ['sensitive_personal_data'],
                'encryption_required': True
            },
            ComplianceRegion.CCPA: {
                'data_retention_days': 365 * 2,  # 2 years  
                'anonymization_required': True,
                'consent_required': False,  # Opt-out rather than opt-in
                'right_to_erasure': True,
                'data_portability': True,
                'privacy_by_design': False,
                'sale_notification_required': True,
                'allowed_transfers': ['us_domestic', 'adequacy'],
                'prohibited_data_types': [],
                'encryption_required': True
            },
            ComplianceRegion.PDPA: {
                'data_retention_days': 365 * 3,  # 3 years
                'anonymization_required': True, 
                'consent_required': True,
                'right_to_erasure': True,
                'data_portability': True,
                'privacy_by_design': True,
                'data_localization': True,  # Singapore/Thailand requirements
                'allowed_transfers': ['adequacy_decision', 'consent'],
                'prohibited_data_types': ['sensitive_personal_data'],
                'encryption_required': True
            },
            ComplianceRegion.PIPEDA: {
                'data_retention_days': 365 * 3,  # 3 years
                'anonymization_required': True,
                'consent_required': True,
                'right_to_erasure': True,
                'data_portability': True,
                'privacy_by_design': True,
                'breach_notification_72h': True,
                'allowed_transfers': ['adequacy', 'consent', 'contracts'],
                'prohibited_data_types': [],
                'encryption_required': True
            },
            ComplianceRegion.LGPD: {
                'data_retention_days': 365 * 3,  # 3 years
                'anonymization_required': True,
                'consent_required': True,
                'right_to_erasure': True,
                'data_portability': True,
                'privacy_by_design': True,
                'dpo_required': True,
                'allowed_transfers': ['adequacy', 'consent', 'international_cooperation'],
                'prohibited_data_types': ['sensitive_personal_data'],
                'encryption_required': True
            }
        }
    
    def validate_compliance(self, region: ComplianceRegion, operation: str, data_types: List[str] = None) -> Dict[str, Any]:
        """Validate compliance for a specific region and operation.
        
        Args:
            region: Compliance region to validate against
            operation: Type of operation (store, process, transfer, etc.)
            data_types: Types of data involved
            
        Returns:
            Compliance validation result
        """
        settings = self._compliance_settings.get(region)
        if not settings:
            return {
                'compliant': False,
                'reason': f'No compliance settings for region {region}',
                'requirements': []
            }
        
        requirements = []
        compliant = True
        
        # Check data type restrictions
        if data_types:
            prohibited = settings.get('prohibited_data_types', [])
            for data_type in data_types:
                if data_type in prohibited:
                    compliant = False
                    requirements.append(f'Data type {data_type} is prohibited in {region.value}')
        
        # Check encryption requirements
        if settings.get('encryption_required', False):
            requirements.append('Data encryption is required')
        
        # Check consent requirements
        if settings.get('consent_required', False):
            requirements.append('User consent is required')
        
        # Check anonymization requirements  
        if settings.get('anonymization_required', False):
            requirements.append('Data anonymization may be required')
        
        # Check data retention limits
        retention_days = settings.get('data_retention_days', 0)
        if retention_days > 0:
            requirements.append(f'Data retention limited to {retention_days} days')
        
        return {
            'compliant': compliant,
            'region': region.value,
            'operation': operation,
            'requirements': requirements,
            'settings': settings
        }
    
    def get_region_compliance(self, region: str) -> List[ComplianceRegion]:
        """Get compliance regions for a deployment region."""
        return self.config.compliance_by_region.get(region, [])
    
    def validate_data_transfer(self, source_region: str, target_region: str, data_types: List[str] = None) -> Dict[str, Any]:
        """Validate data transfer between regions."""
        source_compliance = self.get_region_compliance(source_region)
        target_compliance = self.get_region_compliance(target_region)
        
        # Check each compliance region
        results = []
        overall_compliant = True
        
        for compliance_region in source_compliance + target_compliance:
            result = self.validate_compliance(compliance_region, 'transfer', data_types)
            results.append(result)
            if not result['compliant']:
                overall_compliant = False
        
        return {
            'compliant': overall_compliant,
            'source_region': source_region,
            'target_region': target_region,
            'data_types': data_types or [],
            'validation_results': results,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


class GlobalDeploymentManager:
    """Manager for global and multi-region deployment."""
    
    def __init__(self, 
                 i18n_manager: Optional[InternationalizationManager] = None,
                 compliance_manager: Optional[ComplianceManager] = None):
        self.i18n = i18n_manager or InternationalizationManager()
        self.compliance = compliance_manager or ComplianceManager()
        self.logger = setup_logger(__name__)
    
    def get_deployment_recommendations(self, user_regions: List[str]) -> Dict[str, Any]:
        """Get deployment recommendations based on user regions."""
        recommendations = {
            'primary_regions': [],
            'secondary_regions': [],
            'compliance_requirements': {},
            'i18n_requirements': set(),
            'cdn_recommendations': {},
            'data_residency_notes': []
        }
        
        # Analyze regions for compliance requirements
        for region in user_regions:
            compliance_regions = self.compliance.get_region_compliance(region)
            recommendations['compliance_requirements'][region] = [cr.value for cr in compliance_regions]
            
            # Recommend primary deployment regions
            if region.startswith('us-'):
                recommendations['primary_regions'].append('us-east-1')
                recommendations['i18n_requirements'].add('en')
            elif region.startswith('eu-'):
                recommendations['primary_regions'].append('eu-west-1')
                recommendations['i18n_requirements'].update(['en', 'fr', 'de', 'es', 'it'])
            elif region.startswith('ap-'):
                recommendations['primary_regions'].append('ap-southeast-1')
                recommendations['i18n_requirements'].update(['en', 'ja', 'zh', 'ko'])
        
        # Convert sets to lists for JSON serialization
        recommendations['i18n_requirements'] = list(recommendations['i18n_requirements'])
        recommendations['primary_regions'] = list(set(recommendations['primary_regions']))
        
        return recommendations
    
    def validate_global_deployment(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a global deployment configuration."""
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'compliance_status': {},
            'i18n_status': {},
            'recommendations': []
        }
        
        # Validate compliance for each region
        for region in deployment_config.get('regions', []):
            compliance_regions = self.compliance.get_region_compliance(region)
            for compliance_region in compliance_regions:
                result = self.compliance.validate_compliance(
                    compliance_region, 
                    'deploy',
                    deployment_config.get('data_types', [])
                )
                validation_results['compliance_status'][f"{region}_{compliance_region.value}"] = result
                
                if not result['compliant']:
                    validation_results['valid'] = False
                    validation_results['issues'].extend(result['requirements'])
        
        # Validate i18n requirements
        required_languages = deployment_config.get('languages', ['en'])
        supported_languages = self.i18n.get_supported_languages()
        
        for lang in required_languages:
            if lang not in supported_languages:
                validation_results['warnings'].append(f"Language {lang} not fully supported")
        
        validation_results['i18n_status'] = {
            'required_languages': required_languages,
            'supported_languages': supported_languages,
            'missing_languages': [l for l in required_languages if l not in supported_languages]
        }
        
        return validation_results


# Global instances
default_i18n_manager = InternationalizationManager()
default_compliance_manager = ComplianceManager()
default_deployment_manager = GlobalDeploymentManager(default_i18n_manager, default_compliance_manager)