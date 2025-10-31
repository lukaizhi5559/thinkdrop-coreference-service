"""
Coreference Resolution using spaCy + neuralcoref
"""

import os
import logging
import spacy
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class CoreferenceResolver:
    """
    Resolves coreferences (pronouns and references) in text using NLP
    """
    
    def __init__(self):
        self.nlp = None
        self.use_neuralcoref = os.getenv('USE_NEURALCOREF', 'true').lower() == 'true'
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize spaCy model with neuralcoref"""
        try:
            model_name = os.getenv('SPACY_MODEL', 'en_core_web_sm')
            logger.info(f"🔄 Loading spaCy model: {model_name}")
            
            self.nlp = spacy.load(model_name)
            
            if self.use_neuralcoref:
                try:
                    import neuralcoref
                    neuralcoref.add_to_pipe(self.nlp)
                    logger.info("✅ neuralcoref added to pipeline")
                except ImportError:
                    logger.warning("⚠️ neuralcoref not available, using spaCy only")
                    self.use_neuralcoref = False
            
            logger.info("✅ Coreference resolver initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
            raise
    
    def resolve(
        self,
        message: str,
        conversation_history: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Resolve coreferences in a message using conversation history
        
        Args:
            message: The current message to resolve
            conversation_history: List of previous messages
            options: Additional options for resolution
        
        Returns:
            Dict with originalMessage, resolvedMessage, replacements, method
        """
        options = options or {}
        
        try:
            # Build full context document
            context_parts = []
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                context_parts.append(msg['content'])
            context_parts.append(message)
            
            full_text = " ".join(context_parts)
            
            # Process with spaCy
            doc = self.nlp(full_text)
            
            # Get resolved text
            if self.use_neuralcoref and hasattr(doc._, 'coref_resolved'):
                resolved_full = doc._.coref_resolved
                method = "neuralcoref"
            else:
                # Fallback to rule-based resolution
                resolved_full = self._rule_based_resolution(message, conversation_history)
                method = "rule_based"
            
            # Extract just the current message (last part)
            resolved_message = self._extract_current_message(
                resolved_full,
                message,
                len(context_parts) - 1
            )
            
            # Find replacements
            replacements = self._find_replacements(message, resolved_message)
            
            return {
                "originalMessage": message,
                "resolvedMessage": resolved_message,
                "replacements": replacements,
                "method": method
            }
            
        except Exception as e:
            logger.error(f"❌ Resolution error: {e}", exc_info=True)
            # Return original message if resolution fails
            return {
                "originalMessage": message,
                "resolvedMessage": message,
                "replacements": [],
                "method": "fallback"
            }
    
    def _rule_based_resolution(
        self,
        message: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Simple rule-based coreference resolution
        """
        resolved = message
        
        # Pattern: "the [noun]" or "that [noun]"
        import re
        reference_pattern = r'\b(the|that|this)\s+([a-z]+(?:\s+[a-z]+)?)\b'
        
        matches = list(re.finditer(reference_pattern, message, re.IGNORECASE))
        
        for match in matches:
            reference = match.group(0)
            noun = match.group(2)
            
            # Find antecedent in conversation history
            antecedent = self._find_antecedent(noun, conversation_history)
            
            if antecedent:
                resolved = resolved.replace(reference, antecedent, 1)
        
        # Handle pronouns: he, she, it, they
        pronoun_pattern = r'\b(he|she|it|they|him|her|his|their)\b'
        pronoun_matches = list(re.finditer(pronoun_pattern, message, re.IGNORECASE))
        
        for match in pronoun_matches:
            pronoun = match.group(0)
            antecedent = self._find_pronoun_antecedent(conversation_history)
            
            if antecedent:
                resolved = resolved.replace(pronoun, antecedent, 1)
        
        return resolved
    
    def _find_antecedent(
        self,
        noun: str,
        conversation_history: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        Find the antecedent (what the noun refers to) in conversation history
        """
        import re
        
        # Look for proper nouns (capitalized words) in recent messages
        for msg in reversed(conversation_history[-5:]):
            # Pattern: Capitalized words (proper nouns)
            proper_noun_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
            matches = re.findall(proper_noun_pattern, msg['content'])
            
            if matches:
                # Return the first proper noun found
                return matches[0]
        
        return None
    
    def _find_pronoun_antecedent(
        self,
        conversation_history: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        Find the antecedent for pronouns (he, she, it, they)
        """
        import re
        
        # Look for named entities or proper nouns in recent messages
        for msg in reversed(conversation_history[-3:]):
            # Pattern: Capitalized words
            proper_noun_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
            matches = re.findall(proper_noun_pattern, msg['content'])
            
            if matches:
                return matches[0]
        
        return None
    
    def _extract_current_message(
        self,
        resolved_full: str,
        original_message: str,
        message_index: int
    ) -> str:
        """
        Extract the resolved version of the current message from full text
        """
        # Split by sentences and get the last one(s) that match the original length
        sentences = resolved_full.split(". ")
        
        # Simple heuristic: take the last sentence(s) that roughly match original length
        if len(sentences) > 0:
            return sentences[-1].strip()
        
        return resolved_full
    
    def _find_replacements(
        self,
        original: str,
        resolved: str
    ) -> List[Dict[str, Any]]:
        """
        Find what was replaced in the resolution
        """
        import difflib
        
        replacements = []
        
        # Use difflib to find differences
        matcher = difflib.SequenceMatcher(None, original, resolved)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                replacements.append({
                    "original": original[i1:i2],
                    "resolved": resolved[j1:j2],
                    "confidence": 0.8,  # Placeholder confidence
                    "position": i1
                })
        
        return replacements
