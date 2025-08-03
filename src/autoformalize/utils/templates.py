"""Template management for code generation.

This module provides template management functionality for generating
formal proofs in different proof assistant systems.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass

from ..core.exceptions import ConfigurationError
from .logging_config import setup_logger


@dataclass
class Template:
    """Represents a code generation template."""
    name: str
    content: str
    description: Optional[str] = None
    variables: Dict[str, str] = None
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = {}


class TemplateManager:
    """Manages templates for code generation.
    
    This class handles loading, storing, and retrieving templates
    for different proof assistant systems and statement types.
    """
    
    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize template manager.
        
        Args:
            templates_dir: Directory containing template files
        """
        self.logger = setup_logger(__name__)
        self.templates: Dict[str, Template] = {}
        
        # Load built-in templates
        self._load_builtin_templates()
        
        # Load custom templates if directory provided
        if templates_dir and templates_dir.exists():
            self._load_templates_from_directory(templates_dir)
    
    def _load_builtin_templates(self) -> None:
        """Load built-in templates for common proof assistant systems."""
        
        # Lean 4 templates
        self.templates["lean4_theorem"] = Template(
            name="lean4_theorem",
            content="""-- {description}
theorem {name} {params} : {statement} := by
  {proof_tactics}
""",
            description="Template for Lean 4 theorems",
            variables={
                "name": "Theorem name",
                "params": "Parameters and assumptions",
                "statement": "Theorem statement",
                "proof_tactics": "Proof tactics",
                "description": "Optional description"
            }
        )
        
        self.templates["lean4_definition"] = Template(
            name="lean4_definition",
            content="""-- {description}
def {name} {params} : {type} :=
  {body}
""",
            description="Template for Lean 4 definitions",
            variables={
                "name": "Definition name",
                "params": "Parameters",
                "type": "Return type",
                "body": "Definition body",
                "description": "Optional description"
            }
        )
        
        self.templates["lean4_lemma"] = Template(
            name="lean4_lemma",
            content="""-- {description}
lemma {name} {params} : {statement} := by
  {proof_tactics}
""",
            description="Template for Lean 4 lemmas",
            variables={
                "name": "Lemma name",
                "params": "Parameters and assumptions",
                "statement": "Lemma statement",
                "proof_tactics": "Proof tactics",
                "description": "Optional description"
            }
        )
        
        # Isabelle templates
        self.templates["isabelle_theorem"] = Template(
            name="isabelle_theorem",
            content="""(* {description} *)
theorem {name}: "{statement}"
proof {proof_method}
  {proof_steps}
qed
""",
            description="Template for Isabelle theorems",
            variables={
                "name": "Theorem name",
                "statement": "Theorem statement",
                "proof_method": "Proof method (e.g., -)",
                "proof_steps": "Proof steps",
                "description": "Optional description"
            }
        )
        
        self.templates["isabelle_definition"] = Template(
            name="isabelle_definition",
            content="""(* {description} *)
definition {name} :: "{type}" where
  "{name} ≡ {body}"
""",
            description="Template for Isabelle definitions",
            variables={
                "name": "Definition name",
                "type": "Type signature",
                "body": "Definition body",
                "description": "Optional description"
            }
        )
        
        # Coq templates
        self.templates["coq_theorem"] = Template(
            name="coq_theorem",
            content="""(* {description} *)
Theorem {name} : {statement}.
Proof.
  {proof_tactics}
Qed.
""",
            description="Template for Coq theorems",
            variables={
                "name": "Theorem name",
                "statement": "Theorem statement",
                "proof_tactics": "Proof tactics",
                "description": "Optional description"
            }
        )
        
        self.templates["coq_definition"] = Template(
            name="coq_definition",
            content="""(* {description} *)
Definition {name} {params} : {type} :=
  {body}.
""",
            description="Template for Coq definitions",
            variables={
                "name": "Definition name",
                "params": "Parameters",
                "type": "Return type",
                "body": "Definition body",
                "description": "Optional description"
            }
        )
        
        # Generic prompt templates
        self.templates["formalization_prompt"] = Template(
            name="formalization_prompt",
            content="""Convert the following mathematical {statement_type} to {target_system} code.

{statement_type.title()}: {name}
Statement: {statement}
{proof_section}

Please provide:
1. Proper {target_system} syntax and formatting
2. Appropriate imports and dependencies
3. Complete and correct proof tactics
4. Clear variable declarations and types

Generate only valid {target_system} code with minimal comments.
""",
            description="Generic formalization prompt template",
            variables={
                "statement_type": "Type of statement (theorem, lemma, etc.)",
                "target_system": "Target proof assistant system",
                "name": "Statement name",
                "statement": "Mathematical statement",
                "proof_section": "Proof content (if available)"
            }
        )
        
        self.logger.info(f"Loaded {len(self.templates)} built-in templates")
    
    def _load_templates_from_directory(self, templates_dir: Path) -> None:
        """Load templates from a directory.
        
        Args:
            templates_dir: Directory containing template files
        """
        try:
            for template_file in templates_dir.glob("*.json"):
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                
                template = Template(
                    name=template_data["name"],
                    content=template_data["content"],
                    description=template_data.get("description"),
                    variables=template_data.get("variables", {})
                )
                
                self.templates[template.name] = template
                self.logger.debug(f"Loaded template: {template.name}")
            
            self.logger.info(f"Loaded custom templates from {templates_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load templates from {templates_dir}: {e}")
    
    def get_template(self, name: str) -> Optional[str]:
        """Get template content by name.
        
        Args:
            name: Template name
            
        Returns:
            Template content string or None if not found
        """
        template = self.templates.get(name)
        return template.content if template else None
    
    def get_template_object(self, name: str) -> Optional[Template]:
        """Get full template object by name.
        
        Args:
            name: Template name
            
        Returns:
            Template object or None if not found
        """
        return self.templates.get(name)
    
    def add_template(self, template: Template) -> None:
        """Add a new template.
        
        Args:
            template: Template object to add
        """
        self.templates[template.name] = template
        self.logger.debug(f"Added template: {template.name}")
    
    def remove_template(self, name: str) -> bool:
        """Remove a template by name.
        
        Args:
            name: Template name to remove
            
        Returns:
            True if template was removed, False if not found
        """
        if name in self.templates:
            del self.templates[name]
            self.logger.debug(f"Removed template: {name}")
            return True
        return False
    
    def list_templates(self) -> Dict[str, str]:
        """List all available templates.
        
        Returns:
            Dictionary mapping template names to descriptions
        """
        return {
            name: template.description or "No description"
            for name, template in self.templates.items()
        }
    
    def render_template(self, name: str, **kwargs) -> Optional[str]:
        """Render a template with provided variables.
        
        Args:
            name: Template name
            **kwargs: Template variables
            
        Returns:
            Rendered template string or None if template not found
        """
        template = self.templates.get(name)
        if not template:
            return None
        
        try:
            return template.content.format(**kwargs)
        except KeyError as e:
            self.logger.warning(f"Missing variable {e} for template {name}")
            return template.content  # Return unrendered template
        except Exception as e:
            self.logger.error(f"Failed to render template {name}: {e}")
            return None
    
    def validate_template(self, name: str, **kwargs) -> Dict[str, Any]:
        """Validate template variables.
        
        Args:
            name: Template name
            **kwargs: Variables to validate
            
        Returns:
            Validation result dictionary
        """
        template = self.templates.get(name)
        if not template:
            return {"valid": False, "error": f"Template {name} not found"}
        
        missing_vars = []
        extra_vars = []
        
        # Check for missing required variables
        for var in template.variables:
            if var not in kwargs:
                missing_vars.append(var)
        
        # Check for extra variables
        for var in kwargs:
            if var not in template.variables:
                extra_vars.append(var)
        
        return {
            "valid": len(missing_vars) == 0,
            "missing_variables": missing_vars,
            "extra_variables": extra_vars,
            "required_variables": list(template.variables.keys())
        }
    
    def save_template(self, template: Template, filepath: Path) -> None:
        """Save a template to a JSON file.
        
        Args:
            template: Template to save
            filepath: File path to save to
        """
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            template_data = {
                "name": template.name,
                "content": template.content,
                "description": template.description,
                "variables": template.variables
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved template {template.name} to {filepath}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save template: {e}")
    
    def load_template_from_file(self, filepath: Path) -> Template:
        """Load a template from a JSON file.
        
        Args:
            filepath: Path to template file
            
        Returns:
            Loaded template object
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
            
            template = Template(
                name=template_data["name"],
                content=template_data["content"],
                description=template_data.get("description"),
                variables=template_data.get("variables", {})
            )
            
            self.add_template(template)
            return template
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load template from {filepath}: {e}")
    
    def export_templates(self, output_dir: Path) -> None:
        """Export all templates to JSON files.
        
        Args:
            output_dir: Directory to export templates to
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for template in self.templates.values():
            filepath = output_dir / f"{template.name}.json"
            self.save_template(template, filepath)
        
        self.logger.info(f"Exported {len(self.templates)} templates to {output_dir}")
    
    def create_custom_template(
        self,
        name: str,
        content: str,
        description: Optional[str] = None,
        **variables
    ) -> Template:
        """Create and add a custom template.
        
        Args:
            name: Template name
            content: Template content with format placeholders
            description: Optional description
            **variables: Variable descriptions
            
        Returns:
            Created template object
        """
        template = Template(
            name=name,
            content=content,
            description=description,
            variables=variables
        )
        
        self.add_template(template)
        return template


# Global template manager instance
default_template_manager = TemplateManager()
