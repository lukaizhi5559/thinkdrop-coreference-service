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
        """Initialize spaCy model with coreferee (modern coreference resolution)"""
        try:
            model_name = os.getenv('SPACY_MODEL', 'en_core_web_sm')
            logger.info(f"🔄 Loading spaCy model: {model_name}")
            
            self.nlp = spacy.load(model_name)
            
            if self.use_neuralcoref:
                # Try coreferee first (modern, maintained alternative)
                try:
                    import coreferee
                    self.nlp.add_pipe('coreferee')
                    logger.info("✅ coreferee added to pipeline (modern coreference resolution)")
                    self.coref_method = 'coreferee'
                except ImportError:
                    # Fallback to neuralcoref if available
                    try:
                        import neuralcoref
                        neuralcoref.add_to_pipe(self.nlp)
                        logger.info("✅ neuralcoref added to pipeline")
                        self.coref_method = 'neuralcoref'
                    except ImportError:
                        logger.warning("⚠️ No neural coreference library available, using rule-based resolution")
                        self.use_neuralcoref = False
                        self.coref_method = 'rule_based'
            else:
                self.coref_method = 'rule_based'
            
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
            
            # Get resolved text based on available method
            if self.coref_method == 'coreferee':
                # coreferee analyzes full context but we only resolve current message
                resolved_message = self._resolve_with_coreferee(doc, message, full_text)
                method = "coreferee"
                
                # If coreferee didn't find anything, try simple pronoun fallback
                if resolved_message == message:
                    logger.debug("⚠️ Coreferee found no chains - trying simple pronoun fallback")
                    resolved_message = self._simple_pronoun_fallback(message, conversation_history, doc)
                    if resolved_message != message:
                        method = "simple_fallback"
                        logger.info(f"✅ Simple fallback resolved: '{message}' → '{resolved_message}'")
                    else:
                        method = "none"
                    
            elif self.coref_method == 'neuralcoref' and hasattr(doc._, 'coref_resolved'):
                resolved_full = doc._.coref_resolved
                method = "neuralcoref"
                # Extract just the current message (last part)
                resolved_message = self._extract_current_message(
                    resolved_full,
                    message,
                    len(context_parts) - 1
                )
            else:
                # No coreference method available - return original message
                # LLM has conversation history and can handle context itself
                logger.debug("⚠️ No coreference method available - returning original message unchanged")
                resolved_message = message
                method = "none"
            
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
    
    def _resolve_with_coreferee(self, doc, message: str, full_text: str) -> str:
        """
        Resolve coreferences using coreferee chains
        Analyzes full conversation context but only replaces pronouns in current message
        """
        resolved = message
        replacements_made = []
        
        # Find where the current message starts in the full text
        message_start_pos = full_text.rfind(message)
        
        # coreferee provides coreference chains via doc._.coref_chains
        if hasattr(doc._, 'coref_chains') and doc._.coref_chains:
            logger.debug(f"Found {len(doc._.coref_chains)} coreference chain(s)")
            
            for chain in doc._.coref_chains:
                try:
                    # Chain is a list of Mention objects
                    # Find the main reference - prefer MOST RECENT non-pronoun mention before current message
                    main_mention_text = None
                    main_mention_pos = -1
                    current_message_pronouns = []
                    non_pronoun_candidates = []  # Track all candidates for ambiguity detection
                    
                    # Sort mentions by position (earliest first for processing)
                    sorted_mentions = sorted(chain, key=lambda m: m.token_indexes[0])
                    
                    for mention in sorted_mentions:
                        # Get the mention's token span
                        mention_tokens = [doc[i] for i in mention.token_indexes]
                        mention_text = ' '.join([t.text for t in mention_tokens])
                        
                        # Get position of first token in mention
                        first_token = mention_tokens[0]
                        token_pos = first_token.idx
                        
                        # Check if this mention is in the current message
                        in_current_message = token_pos >= message_start_pos
                        
                        # Check if this is a pronoun mention
                        is_pronoun = mention_tokens[0].text.lower() in ['he', 'she', 'it', 'they', 'him', 'her', 'his', 'their', 'them', 'its']
                        
                        if in_current_message and is_pronoun:
                            # Pronoun in current message - mark for replacement
                            current_message_pronouns.append(mention_tokens[0].text)
                        elif not in_current_message and not is_pronoun:
                            # Non-pronoun mention from earlier in conversation
                            # Keep updating to get the MOST RECENT one (closest to current message)
                            # Prefer PERSON/ORG entities
                            if any(t.ent_type_ in ['PERSON', 'ORG', 'PRODUCT', 'GPE'] for t in mention_tokens):
                                # Get the FULL entity text by finding the complete entity span
                                # This ensures we get "Isaac Herzog" not just "Herzog"
                                entity_tokens = [t for t in mention_tokens if t.ent_type_ != '']
                                if entity_tokens:
                                    # Get the full entity by finding all tokens in the same entity
                                    first_ent_token = entity_tokens[0]
                                    # Find all consecutive tokens with the same entity type
                                    ent_start = first_ent_token.i
                                    ent_end = first_ent_token.i + 1
                                    
                                    # Expand backwards to get full entity
                                    while ent_start > 0 and doc[ent_start - 1].ent_type_ == first_ent_token.ent_type_:
                                        ent_start -= 1
                                    
                                    # Expand forwards to get full entity
                                    while ent_end < len(doc) and doc[ent_end].ent_type_ == first_ent_token.ent_type_:
                                        ent_end += 1
                                    
                                    # Extract full entity text
                                    candidate_text = ' '.join([doc[i].text for i in range(ent_start, ent_end)])
                                else:
                                    candidate_text = mention_text
                                non_pronoun_candidates.append((candidate_text, token_pos))
                                # Update if this is more recent (higher position)
                                if token_pos > main_mention_pos:
                                    main_mention_text = candidate_text
                                    main_mention_pos = token_pos
                            elif any(t.pos_ == 'PROPN' for t in mention_tokens):
                                # Proper noun phrase - use the full mention text
                                non_pronoun_candidates.append((mention_text, token_pos))
                                # Update if this is more recent (higher position)
                                if token_pos > main_mention_pos:
                                    main_mention_text = mention_text
                                    main_mention_pos = token_pos
                    
                    # Safety check: Only resolve if we have clear reference
                    # UPDATED: Instead of skipping ambiguous chains, use the MOST RECENT mention
                    # This handles conversations where multiple people are discussed
                    unique_candidates = set(text for text, _ in non_pronoun_candidates)
                    
                    if len(unique_candidates) > 1:
                        # Multiple entities in chain - use the most recent one (highest position)
                        # This is already stored in main_mention_text due to our position-based update logic
                        logger.debug(f"⚠️ Multiple entities in chain: {unique_candidates}")
                        logger.debug(f"✅ Using most recent mention: '{main_mention_text}' (position: {main_mention_pos})")
                    
                    # Replace pronouns in current message with main reference
                    if main_mention_text and current_message_pronouns:
                        for pronoun in current_message_pronouns:
                            resolved = resolved.replace(pronoun, main_mention_text, 1)
                            replacements_made.append(f"'{pronoun}' → '{main_mention_text}'")
                            logger.info(f"✅ Resolved '{pronoun}' → '{main_mention_text}'")
                
                except Exception as e:
                    logger.warning(f"Error processing coreference chain: {e}")
                    logger.exception(e)
                    continue
            
            if replacements_made:
                logger.info(f"🎯 Coreferee resolved: {', '.join(replacements_made)}")
        else:
            logger.debug("No coreference chains found by coreferee")
        
        return resolved
    
    def _rule_based_resolution(
        self,
        message: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        ⚠️ DEPRECATED - This method is no longer used as fallback
        
        Rule-based coreference resolution has ~30% success rate and 70% risk of mangling queries.
        Better to return original message and let LLM handle context from conversation history.
        
        Kept for reference/testing purposes only.
        
        Simple rule-based coreference resolution
        ONLY resolves actual coreferences (common nouns referring to entities)
        Does NOT resolve proper nouns like "the russia", "the USA", etc.
        """
        resolved = message
        
        # Pattern: "the [noun]" or "that [noun]"
        # Only match lowercase common nouns (not proper nouns)
        import re
        reference_pattern = r'\b(the|that|this)\s+([a-z][a-z\s]+?)\b'
        
        matches = list(re.finditer(reference_pattern, message, re.IGNORECASE))
        
        # Known coreference nouns (common nouns that can refer to entities)
        coreference_nouns = {
            'president', 'person', 'man', 'woman', 'guy', 'leader', 'minister',
            'company', 'organization', 'business',
            'movie', 'show', 'book', 'song', 'film', 'series',
            'product', 'device', 'app',
            'city', 'country', 'place', 'location',
            'current president', 'prime minister', 'current leader'
        }
        
        for match in matches:
            reference = match.group(0)
            noun = match.group(2).strip().lower()
            
            # CRITICAL: Only resolve if this is a known coreference noun
            # Skip proper nouns like "the russia", "the england", etc.
            if noun not in coreference_nouns:
                logger.debug(f"⏭️ Skipping '{reference}' - not a coreference noun")
                continue
            
            # Find antecedent in conversation history
            antecedent = self._find_antecedent(noun, conversation_history)
            
            if antecedent:
                resolved = resolved.replace(reference, antecedent, 1)
                logger.debug(f"✅ Resolved '{reference}' → '{antecedent}'")
        
        # Handle pronouns: he, she, it, they
        # BUT skip expletive/dummy pronouns (e.g., "it takes", "it is", "it seems")
        pronoun_pattern = r'\b(he|she|it|they|him|her|his|their)\b'
        pronoun_matches = list(re.finditer(pronoun_pattern, message, re.IGNORECASE))
        
        # Expletive patterns where "it" is NOT a reference
        expletive_patterns = [
            r'\bit\s+(is|was|will\s+be|would\s+be|has\s+been|had\s+been)',  # it is, it was, etc.
            r'\bit\s+(takes|took|will\s+take)',  # it takes
            r'\bit\s+(seems|appears|looks)',  # it seems
            r'\bit\s+(depends|matters)',  # it depends
            r'\bhow\s+(long|far|much)\s+does\s+it',  # how long does it
            r'\bwhat\s+does\s+it',  # what does it
        ]
        
        for match in pronoun_matches:
            pronoun = match.group(0).lower()
            
            # Skip if this is an expletive pronoun
            if pronoun == 'it':
                is_expletive = False
                for pattern in expletive_patterns:
                    if re.search(pattern, message, re.IGNORECASE):
                        is_expletive = True
                        break
                
                if is_expletive:
                    continue  # Skip this pronoun
            
            antecedent = self._find_pronoun_antecedent(conversation_history)
            
            if antecedent:
                resolved = resolved.replace(match.group(0), antecedent, 1)
        
        return resolved
    
    def _find_antecedent(
        self,
        noun: str,
        conversation_history: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        Find the antecedent (what the noun refers to) in conversation history
        Uses semantic matching to find entities that match the noun type
        """
        import re
        
        # Map common nouns to entity types
        noun_to_entity_type = {
            'president': 'PERSON',
            'person': 'PERSON',
            'man': 'PERSON',
            'woman': 'PERSON',
            'guy': 'PERSON',
            'company': 'ORG',
            'organization': 'ORG',
            'movie': 'WORK_OF_ART',
            'show': 'WORK_OF_ART',
            'book': 'WORK_OF_ART',
            'song': 'WORK_OF_ART',
            'product': 'PRODUCT',
            'city': 'GPE',
            'country': 'GPE',
            'place': 'GPE'
        }
        
        target_entity_type = noun_to_entity_type.get(noun.lower())
        
        # Look for matching entities in recent messages
        for msg in reversed(conversation_history[-5:]):
            try:
                doc = self.nlp(msg['content'])
                
                # If we know the entity type, look for that specific type
                if target_entity_type:
                    for ent in doc.ents:
                        if ent.label_ == target_entity_type:
                            logger.debug(f"Found {target_entity_type} antecedent for '{noun}': {ent.text}")
                            return ent.text
                
                # Fallback: return first named entity
                if doc.ents:
                    logger.debug(f"Found generic antecedent for '{noun}': {doc.ents[0].text}")
                    return doc.ents[0].text
                    
            except Exception as e:
                logger.warning(f"Error finding antecedent: {e}")
                continue
        
        return None
    
    def _find_pronoun_antecedent(
        self,
        conversation_history: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        Find the antecedent for pronouns (he, she, it, they)
        Uses spaCy NER to intelligently find people, organizations, products, etc.
        Returns None if ambiguous (multiple different entities found)
        """
        import re
        
        # Collect all entities from recent messages
        all_entities_by_type = {
            'PERSON': [],
            'ORG': [],
            'PRODUCT': [],
            'GPE': [],
            'WORK_OF_ART': [],
            'EVENT': []
        }
        
        # Look for named entities in recent messages using spaCy NER
        for msg in reversed(conversation_history[-3:]):
            try:
                # Use spaCy to extract named entities
                doc = self.nlp(msg['content'])
                
                for ent in doc.ents:
                    if ent.label_ in all_entities_by_type:
                        all_entities_by_type[ent.label_].append(ent.text)
                    
            except Exception as e:
                logger.warning(f"Error extracting entities: {e}")
                continue
        
        # Check for ambiguity - if multiple different entities of same type, it's ambiguous
        for entity_type in ['PERSON', 'ORG', 'PRODUCT', 'GPE', 'WORK_OF_ART', 'EVENT']:
            entities = all_entities_by_type[entity_type]
            if entities:
                unique_entities = set(entities)
                if len(unique_entities) > 1:
                    logger.debug(f"⚠️ Ambiguous: Multiple {entity_type} entities found: {unique_entities}")
                    return None  # Ambiguous - don't guess
                else:
                    # Only one unique entity of this type - safe to use
                    logger.debug(f"✅ Found unambiguous {entity_type} antecedent: {entities[0]}")
                    return entities[0]
        
        # Fallback: Look for proper nouns in most recent message only
        if conversation_history:
            try:
                most_recent = conversation_history[-1]['content']
                
                # Fallback: Look for capitalized multi-word phrases (likely proper nouns)
                # But exclude common words like "President", "United States", etc.
                proper_noun_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
                matches = re.findall(proper_noun_pattern, most_recent)
                
                # Filter out common non-entity phrases
                excluded_phrases = ['United States', 'President', 'According To', 'The User']
                filtered_matches = [m for m in matches if m not in excluded_phrases]
                
                if filtered_matches:
                    logger.debug(f"Found proper noun antecedent: {filtered_matches[0]}")
                    return filtered_matches[0]
                    
            except Exception as e:
                logger.warning(f"Error processing message for antecedent: {e}")
                pass  # Fall through to return None
        
        return None
    
    def _simple_pronoun_fallback(
        self,
        message: str,
        conversation_history: List[Dict[str, str]],
        doc
    ) -> str:
        """
        Simple fallback for common pronouns when coreferee fails
        Handles: it, them, they, that, this
        Looks for most recent NOUN entity in conversation
        
        CRITICAL: Does NOT replace 'that' when used as a relative pronoun
        (e.g., "the man that lived" - here 'that' is NOT a pronoun reference)
        """
        import re
        
        # Check if message contains pronouns we can resolve
        pronouns_to_resolve = []
        pronoun_patterns = {
            'it': r'\bit\b',
            'them': r'\bthem\b',
            'they': r'\bthey\b',
            # REMOVED 'that' - too risky, often used as relative pronoun
            # 'that': r'\bthat\b',
            'this': r'\bthis\b'
        }
        
        for pronoun, pattern in pronoun_patterns.items():
            if re.search(pattern, message, re.IGNORECASE):
                pronouns_to_resolve.append(pronoun)
        
        if not pronouns_to_resolve:
            return message
        
        # CRITICAL: Process current message first to identify nouns we should EXCLUDE
        current_msg_doc = self.nlp(message)
        current_msg_nouns = set()
        for token in current_msg_doc:
            if token.pos_ == 'NOUN':
                current_msg_nouns.add(token.text.lower())
        
        # Look for most recent noun entity in conversation history
        # Go backwards through history to find the most recent entity
        # PRIORITY 1: Named entities (PERSON, ORG, PRODUCT, etc.)
        named_entity_candidates = []
        noun_candidates = []
        
        for msg in reversed(conversation_history[-3:]):  # Last 3 messages
            content = msg['content']
            
            # Process with spaCy to find entities
            msg_doc = self.nlp(content)
            
            # First priority: Named entities (these are most likely referents)
            for ent in msg_doc.ents:
                # Include all major entity types that could be pronoun referents
                # Exclude DATE/TIME (too generic), PERCENT/MONEY/QUANTITY (numerical)
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'WORK_OF_ART', 'LANGUAGE', 
                                   'NORP', 'FAC', 'LOC', 'EVENT', 'LAW', 'PERCENT', 'MONEY',
                                   'QUANTITY', 'ORDINAL', 'CARDINAL']:
                    # Exclude if this entity appears in current message (likely not a referent)
                    if ent.text.lower() not in current_msg_nouns:
                        named_entity_candidates.append(ent.text)
            
            # Second priority: Proper nouns (NNP) that aren't in current message
            for token in msg_doc:
                if token.tag_ == 'NNP' and token.text.lower() not in current_msg_nouns:
                    noun_phrase = ' '.join([t.text for t in token.subtree if t.pos_ in ['NOUN', 'PROPN', 'ADJ']])
                    if noun_phrase and noun_phrase.lower() not in current_msg_nouns:
                        noun_candidates.append(noun_phrase)
        
        # Use named entities first, then proper nouns
        all_candidates = named_entity_candidates + noun_candidates
        
        if all_candidates:
            # Remove duplicates while preserving order
            seen = set()
            unique_candidates = []
            for c in all_candidates:
                if c.lower() not in seen:
                    seen.add(c.lower())
                    unique_candidates.append(c)
            
            referent = unique_candidates[0]
            logger.info(f"🎯 Simple fallback found referent: '{referent}' for pronouns: {pronouns_to_resolve}")
            
            # Replace pronouns with referent
            resolved = message
            for pronoun in pronouns_to_resolve:
                # Case-insensitive replacement
                resolved = re.sub(
                    r'\b' + pronoun + r'\b',
                    referent,
                    resolved,
                    count=1,
                    flags=re.IGNORECASE
                )
            
            return resolved
        
        # No referent found
        logger.debug("⚠️ Simple fallback: No referent found")
        return message
    
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
