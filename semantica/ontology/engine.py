from typing import Any, Dict, List, Optional

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker
from .ontology_generator import OntologyGenerator
from .class_inferrer import ClassInferrer
from .property_generator import PropertyGenerator
from .owl_generator import OWLGenerator
from .ontology_evaluator import OntologyEvaluator
from .ontology_validator import OntologyValidator
from .llm_generator import LLMOntologyGenerator


class OntologyEngine:
    def __init__(self, **config):
        self.logger = get_logger("ontology_engine")
        self.progress = get_progress_tracker()
        self.config = config

        self.generator = OntologyGenerator(**config)
        self.inferrer = ClassInferrer(**config)
        self.propgen = PropertyGenerator(**config)
        self.owl = OWLGenerator(**config)
        self.evaluator = OntologyEvaluator(**config)
        self.validator = OntologyValidator(**config)
        self.llm = LLMOntologyGenerator(**config)
        self.store = config.get("store")
        
        from ..change_management.ontology_version_manager import VersionManager
        self.version_manager = config.get("version_manager") or VersionManager(**config)

    def from_data(self, data: Dict[str, Any], **options) -> Dict[str, Any]:
        tracking_id = self.progress.start_tracking(
            module="ontology",
            submodule="OntologyEngine",
            message="Generating ontology from data",
        )
        try:
            ontology = self.generator.generate_ontology(data, **options)
            self.progress.update_tracking(tracking_id, message="Ontology generated")
            return ontology
        except Exception as e:
            self.progress.update_tracking(tracking_id, message="Generation failed")
            raise

    def from_text(self, text: str, provider: Optional[str] = None, model: Optional[str] = None, **options) -> Dict[str, Any]:
        if provider:
            self.llm.set_provider(provider, model=model)
        return self.llm.generate_ontology_from_text(text, **options)

    def infer_classes(self, entities: List[Dict[str, Any]], **options) -> List[Dict[str, Any]]:
        return self.inferrer.infer_classes(entities, **options)

    def infer_properties(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        classes: List[Dict[str, Any]],
        **options,
    ) -> List[Dict[str, Any]]:
        return self.propgen.infer_properties(entities, relationships, classes, **options)

    def evaluate(self, ontology: Dict[str, Any], **options):
        return self.evaluator.evaluate_ontology(ontology, **options)

    def validate(self, ontology: Dict[str, Any], **options):
        return self.validator.validate(ontology, **options)

    def to_owl(self, ontology: Dict[str, Any], format: str = "turtle", **options):
        return self.owl.generate_owl(ontology, format=format, **options)

    def export_owl(self, ontology: Dict[str, Any], path: str, format: str = "turtle"):
        return self.owl.export_owl(ontology, path, format=format)
    
    def get_ontology_version_dict(self, version_id: str) -> Dict[str, Any]:
        """ Utility to load an ontology version as plain dict ready for diffing."""
        
        version_record = self.version_manager.get_version(version_id)
        if not version_record:
            raise ProcessingError(f"Version {version_id} not found.")
        
        return version_record.metadata.get("structure", {"classes": [], "properties": []})
    
    def compare_versions(self, base_id: str, target_id: str, **options) -> Dict[str, Any]:
        """
        Orchestrates version loading, diff computation, and report generation.
        
        Args:
            base_id: Version ID of the old ontology
            target_id: Version ID of the new ontology
            **options: Can pass 'base_dict' and 'target_dict' directly to bypass loading.
            
        Returns:
             A structured ImpactReport dictionary containing breaking/safe changes.
        """
        
        tracking_id = self.progress.start_tracking(
            module="ontology",
            submodule="OntologyEngine",
            message=f"Comparing ontology versions: {base_id} -> {target_id}"
        )
        
        try:
            from ..change_management.change_log import generate_change_report
            
            base_dict = options.get("base_dict") or self.get_ontology_version_dict(base_id)
            target_dict = options.get("target_dict") or self.get_ontology_version_dict(target_id)
            
            diff_result = self.version_manager.diff_ontologies(base_dict, target_dict)
            
            report = generate_change_report(diff_result)
            
            if options.get("run_validation"):
                self.progress.update_tracking(tracking_id, message="Running validation on target schema...")
                
                validation_results = self.validate(target_dict, **options)
                
                report["validation_results"] = validation_results
            
            self.progress.stop_tracking(tracking_id, status="completed", message="Comparison complete")
            
            return report
        
        except Exception as e:
            self.progress.stop_tracking(tracking_id, status="failed", message=str(e))
            self.logger.error(f"Failed to compare versions: {e}")
            raise ProcessingError(f"Version comparison failed: {e}")