"""
Coreference Resolution using spaCy + neuralcoref + Context Tracking
"""

import os
import logging
import spacy
from typing import List, Dict, Any, Optional

from .conversation_context import ConversationContext

logger = logging.getLogger(__name__)

class CoreferenceResolver:
    """
    Resolves coreferences (pronouns and references) in text using NLP
    Uses ConversationContext for semantic understanding and state tracking
    """
    
    def __init__(self):
        self.nlp = None
        self.use_neuralcoref = os.getenv('USE_NEURALCOREF', 'true').lower() == 'true'
        self.context = None  # Will be initialized after nlp model loads
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
            
            # Initialize conversation context tracker
            self.context = ConversationContext(self.nlp)
            
            logger.info("✅ Coreference resolver initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
            raise
    
    def _rebuild_context_from_history(self, conversation_history: List[Dict[str, str]]) -> None:
        """
        Rebuild conversation context from history
        This allows the context tracker to understand the full conversation state
        """
        # Reset context
        self.context = ConversationContext(self.nlp)
        
        # Process each message to build up context
        for i, msg in enumerate(conversation_history):
            if msg.get('role') == 'user':
                self.context.update_from_user_message(msg['content'], i)
            elif msg.get('role') == 'assistant':
                # Extract entities from AI responses for pronoun resolution
                self.context.update_from_assistant_message(msg['content'], i)
        
        # Log current context state
        if self.context.current_topic:
            logger.debug(f"📊 Context state: {self.context.get_context_summary()}")
    
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
        
        # Extract screen content from options (for screen_intelligence intents)
        screen_content = options.get('screenContent')
        intent_type = options.get('intentType')
        
        if screen_content:
            logger.info(f"🖥️  Screen content provided for {intent_type} intent")
            logger.debug(f"   Screen content preview: '{screen_content[:150]}...'")
        
        try:
            # Rebuild conversation context from history
            # This allows the context tracker to understand the full conversation state
            self._rebuild_context_from_history(conversation_history)
            
            # Build full context document and track message boundaries
            context_parts = []
            message_roles = []  # Track whether each part is from user or assistant
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                context_parts.append(msg['content'])
                message_roles.append(msg.get('role', 'user'))  # Default to user if role not specified
            context_parts.append(message)
            message_roles.append('user')  # Current message is always from user
            
            full_text = " ".join(context_parts)
            
            # Store message boundaries and roles for later use
            self._message_boundaries = []
            current_pos = 0
            for i, part in enumerate(context_parts):
                start_pos = current_pos
                end_pos = start_pos + len(part)
                self._message_boundaries.append({
                    'start': start_pos,
                    'end': end_pos,
                    'role': message_roles[i],
                    'text': part
                })
                current_pos = end_pos + 1  # +1 for the space separator
            
            # Process with spaCy
            doc = self.nlp(full_text)
            
            # EARLY CHECK: Skip resolution for imperative commands
            # Check if the message is a standalone imperative command that doesn't need context
            early_needs_check = self._check_needs_context(message, [], "none")
            if not early_needs_check:
                # This is a standalone command - return it unchanged
                logger.info(f"⏭️  Skipping resolution for standalone imperative command: '{message}'")
                return {
                    "originalMessage": message,
                    "resolvedMessage": message,
                    "replacements": [],
                    "method": "none",
                    "needsContext": False
                }
            
            # Get resolved text based on available method
            if self.coref_method == 'coreferee':
                # coreferee analyzes full context but we only resolve current message
                resolved_message = self._resolve_with_coreferee(doc, message, full_text)
                method = "coreferee"
                
                # If coreferee didn't find anything, try simple pronoun fallback
                if resolved_message == message:
                    logger.debug("⚠️ Coreferee found no chains - trying simple pronoun fallback")
                    resolved_message = self._simple_pronoun_fallback(message, conversation_history, doc, screen_content)
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
            
            # Determine if this query needs conversation context
            # True if: coreferences were found OR query has elliptical reference patterns
            needs_context = self._check_needs_context(message, replacements, method)
            
            return {
                "originalMessage": message,
                "resolvedMessage": resolved_message,
                "replacements": replacements,
                "method": method,
                "needsContext": needs_context
            }
            
        except Exception as e:
            logger.error(f"❌ Resolution error: {e}", exc_info=True)
            # Return original message if resolution fails
            # Still check if context is needed even on error
            needs_context = self._check_needs_context(message, [], "fallback")
            return {
                "originalMessage": message,
                "resolvedMessage": message,
                "replacements": [],
                "method": "fallback",
                "needsContext": needs_context
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
                        elif in_current_message and not is_pronoun:
                            # CRITICAL: Entity in current message (same sentence as pronoun)
                            # This should have HIGHEST priority - use position 10000000 to ensure it wins
                            if any(t.ent_type_ in ['PERSON', 'ORG', 'PRODUCT', 'GPE'] for t in mention_tokens):
                                entity_tokens = [t for t in mention_tokens if t.ent_type_ != '']
                                if entity_tokens:
                                    first_ent_token = entity_tokens[0]
                                    ent_start = first_ent_token.i
                                    ent_end = first_ent_token.i + 1
                                    
                                    while ent_start > 0 and doc[ent_start - 1].ent_type_ == first_ent_token.ent_type_:
                                        ent_start -= 1
                                    while ent_end < len(doc) and doc[ent_end].ent_type_ == first_ent_token.ent_type_:
                                        ent_end += 1
                                    
                                    candidate_text = ' '.join([doc[i].text for i in range(ent_start, ent_end)])
                                else:
                                    candidate_text = mention_text
                                
                                # Use very high position to prioritize current message entities
                                non_pronoun_candidates.append((candidate_text, 10000000, True))
                                main_mention_text = candidate_text
                                main_mention_pos = 10000000
                                logger.info(f"🎯 Prioritizing entity from current message: '{candidate_text}'")
                            elif any(t.pos_ == 'PROPN' for t in mention_tokens):
                                non_pronoun_candidates.append((mention_text, 10000000, True))
                                main_mention_text = mention_text
                                main_mention_pos = 10000000
                                logger.info(f"🎯 Prioritizing proper noun from current message: '{mention_text}'")
                        elif not in_current_message and not is_pronoun:
                            # Non-pronoun mention from earlier in conversation
                            # CRITICAL: Prioritize entities from USER messages over ASSISTANT messages
                            # This prevents picking entities the AI mentioned over entities the user explicitly referenced
                            
                            # Determine if this mention is from a user or assistant message
                            is_from_user = False
                            if hasattr(self, '_message_boundaries'):
                                for boundary in self._message_boundaries:
                                    if boundary['start'] <= token_pos < boundary['end']:
                                        is_from_user = (boundary['role'] == 'user')
                                        break
                            
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
                                
                                # Prioritize user messages: add 1000000 to position if from user
                                # This ensures user entities always win over assistant entities
                                priority_pos = token_pos + (1000000 if is_from_user else 0)
                                non_pronoun_candidates.append((candidate_text, priority_pos, is_from_user))
                                
                                # Update if this has higher priority
                                if priority_pos > main_mention_pos:
                                    main_mention_text = candidate_text
                                    main_mention_pos = priority_pos
                                    logger.debug(f"📌 Candidate: '{candidate_text}' (pos: {token_pos}, user: {is_from_user}, priority: {priority_pos})")
                            elif any(t.pos_ == 'PROPN' for t in mention_tokens):
                                # Proper noun phrase - use the full mention text
                                priority_pos = token_pos + (1000000 if is_from_user else 0)
                                non_pronoun_candidates.append((mention_text, priority_pos, is_from_user))
                                
                                # Update if this has higher priority
                                if priority_pos > main_mention_pos:
                                    main_mention_text = mention_text
                                    main_mention_pos = priority_pos
                                    logger.debug(f"📌 Candidate: '{mention_text}' (pos: {token_pos}, user: {is_from_user}, priority: {priority_pos})")
                    
                    # Safety check: Only resolve if we have clear reference
                    # UPDATED: Instead of skipping ambiguous chains, use the HIGHEST PRIORITY mention
                    # Priority: current message (10000000) > user messages (1000000+pos) > assistant messages (pos)
                    unique_candidates = set(text for text, *_ in non_pronoun_candidates)
                    
                    if len(unique_candidates) > 1:
                        # Multiple entities in chain - use the highest priority one
                        # This is already stored in main_mention_text due to our priority-based update logic
                        logger.debug(f"⚠️ Multiple entities in chain: {unique_candidates}")
                        logger.info(f"✅ Using highest priority mention: '{main_mention_text}' (priority: {main_mention_pos})")
                    
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
        doc,
        screen_content: Optional[str] = None
    ) -> str:
        """
        Context-aware fallback using ConversationContext for semantic resolution
        
        Handles:
        1. Elliptical patterns (e.g., "second best", "what about X")
        2. Pronouns (it, them, they, this, that)
        3. Short answers to AI questions
        
        Uses ConversationContext for semantic understanding instead of brittle regex patterns
        """
        import re
        
        # Get last assistant message if exists
        last_assistant_message = None
        if conversation_history:
            for msg in reversed(conversation_history):
                if msg.get('role') == 'assistant':
                    last_assistant_message = msg.get('content', '')
                    break
        
        # Try context-aware resolution methods in order of specificity
        
        # 1. Try elliptical resolution (e.g., "what's the second best", "what about X")
        resolved = self.context.resolve_elliptical(message)
        if resolved:
            return resolved
        
        # 2. Try pronoun resolution (e.g., "tell me about it")
        resolved = self.context.resolve_pronoun(message, doc)
        if resolved:
            return resolved
        
        # 3. Try short answer resolution (e.g., "France" after "Which region?")
        if last_assistant_message:
            resolved = self.context.resolve_short_answer(message, last_assistant_message)
            if resolved:
                return resolved
        
        # 4. No resolution found - return original message
        logger.debug("⏭️ No context-aware resolution found, returning original message")
        return message
    def _resolve_elliptical_reference(
        self,
        message: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Resolve elliptical references like "second best", "another one", "the next"
        by extracting the subject from previous conversation context.
        
        E.g., "What's the best lasagna?" → "What's the second best?" 
        Should resolve to: "What's the second best lasagna?"
        """
        import re
        
        if not conversation_history:
            logger.info("⚠️ No conversation history for elliptical resolution")
            return message
        
        # DEBUG: Log conversation history
        logger.debug(f"📋 Conversation history has {len(conversation_history)} messages")
        for i, msg in enumerate(conversation_history):
            logger.debug(f"   [{i}] {msg.get('role')}: {msg.get('content', '')[:50]}...")
        
        # Look at the most recent user message (skip assistant responses)
        # CRITICAL: Skip the first message if it matches the current message (it might be included)
        last_user_message = None
        for msg in reversed(conversation_history):
            if msg.get('role') == 'user':
                # Skip if this is the current message itself
                if msg.get('content', '').strip() != message.strip():
                    last_user_message = msg.get('content', '')
                    break
        
        if not last_user_message:
            logger.info("⚠️ No previous user message found for elliptical resolution")
            return message
        
        logger.info(f"🔍 Analyzing previous user message: '{last_user_message}'")
        
        # SPECIAL CASE: "what about X" / "how about Y" - context-aware topic shift
        # BUT: Check if X itself is an elliptical pattern (e.g., "the second best")
        # vs a new concrete subject (e.g., "cheese cake")
        what_about_match = re.search(r'\b(what about|how about)\s+(.+?)(?:\?|\.\.\.|$)', message, re.IGNORECASE)
        if what_about_match:
            potential_subject = what_about_match.group(2).strip()
            
            # Check if the "subject" is actually an elliptical pattern itself
            # (e.g., "the second best", "the next one", "another option")
            elliptical_subject_patterns = [
                r'^(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(?:best|worst|largest|smallest|biggest|most|least)$',
                r'^(?:the\s+)?(?:next|last|previous|another|other)\s+(?:one|option|choice)$',
            ]
            
            is_elliptical_subject = any(re.match(pattern, potential_subject, re.IGNORECASE) for pattern in elliptical_subject_patterns)
            
            if is_elliptical_subject:
                # This is NOT a topic shift - it's an elliptical reference
                # E.g., "what about the second best" should become "what's the second best [SUBJECT]"
                logger.info(f"🔄 'what about {potential_subject}' is an elliptical reference, not a topic shift")
                # Fall through to regular elliptical resolution below
            else:
                # This IS a topic shift - extract frame and apply to new subject
                # E.g., "what about cheese cake" → "what's the best cheese cake"
                logger.info(f"🔄 Detected topic shift pattern: 'what about {potential_subject}'")
                
                # Extract the query frame from previous message (best/worst/second best/etc.)
                # We want to extract just the ranking part (e.g., "the second best", "the best")
                # not the question word (what's/which)
                frame_pattern = r'(?:what\'?s?|which)\s+((?:the\s+)?(?:(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+)?(?:best|worst|largest|smallest|biggest|most|least))'
                frame_match = re.search(frame_pattern, last_user_message, re.IGNORECASE)
                
                if frame_match:
                    # Extract just the ranking part (group 1), not the question word
                    frame = frame_match.group(1)
                    logger.info(f"✅ Extracted query frame: '{frame}'")
                    
                    # Reconstruct query with new subject
                    resolved = f"what's {frame} {potential_subject}"
                    logger.info(f"✅ Resolved topic shift: '{message}' → '{resolved}'")
                    return resolved
                else:
                    logger.info("⚠️ Could not extract query frame from previous message, treating as new query")
                    return message
        
        # Extract the subject/topic from the previous message
        # Common patterns:
        # - "What's the best [SUBJECT]" → extract SUBJECT
        # - "Tell me about [SUBJECT]" → extract SUBJECT
        # - "Show me [SUBJECT]" → extract SUBJECT
        
        subject_patterns = [
            r'(?:what\'?s?|which|who\'?s?)\s+(?:the\s+)?(?:best|worst|largest|smallest|biggest|most|least)\s+(.+?)(?:\s+in\s+|\s+for\s+|\s+of\s+|\?|\.\.\.|$)',
            r'(?:tell me about|show me|find|search for)\s+(.+?)(?:\?|\.\.\.|$)',
            r'(?:what|which)\s+(.+?)(?:\s+is\s+|\s+are\s+|\?|\.\.\.|$)',
        ]
        
        extracted_subject = None
        for pattern in subject_patterns:
            match = re.search(pattern, last_user_message, re.IGNORECASE)
            if match:
                extracted_subject = match.group(1).strip()
                logger.info(f"✅ Extracted subject from pattern '{pattern}': '{extracted_subject}'")
                break
        
        # If no pattern matched, try spaCy to extract the main noun phrase
        if not extracted_subject:
            logger.info("🔍 No pattern matched, using spaCy to extract subject")
            prev_doc = self.nlp(last_user_message)
            
            # Look for the root noun phrase (subject of the sentence)
            for chunk in prev_doc.noun_chunks:
                # Skip pronouns and articles
                if chunk.root.pos_ in ['NOUN', 'PROPN']:
                    # Clean up: remove leading articles
                    cleaned = chunk.text
                    if cleaned.lower().startswith(('the ', 'a ', 'an ')):
                        cleaned = ' '.join(cleaned.split()[1:])
                    
                    if cleaned and len(cleaned) > 1:
                        extracted_subject = cleaned
                        logger.info(f"✅ Extracted subject from noun chunk: '{extracted_subject}'")
                        break
        
        if not extracted_subject:
            logger.warning("⚠️ Could not extract subject from previous message")
            return message
        
        # Now insert the subject into the current message
        # Patterns to handle:
        # - "what's the second best" → "what's the second best [SUBJECT]"
        # - "show me another one" → "show me another [SUBJECT]"
        # - "the next" → "the next [SUBJECT]"
        
        insertion_patterns = [
            # Pattern: "what about the [ordinal] [superlative]" → convert to "what's the [ordinal] [superlative] [SUBJECT]"
            (r'\b(?:what|how)\s+about\s+((?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(?:best|worst|largest|smallest|biggest|most|least))', r"what's \1 " + extracted_subject),
            # Pattern: "what's the [ordinal] [superlative]" → insert subject after superlative
            (r'(what\'?s?\s+the\s+(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(?:best|worst|largest|smallest|biggest|most|least))', r'\1 ' + extracted_subject),
            # Pattern: "[ordinal] [superlative]" → insert subject after superlative
            (r'((?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(?:best|worst|largest|smallest|biggest|most|least))', r'\1 ' + extracted_subject),
            # Pattern: "another one" → replace "one" with subject
            (r'\banother\s+one\b', 'another ' + extracted_subject),
            # Pattern: "the next" → insert subject after "next"
            (r'\bthe\s+next\b', 'the next ' + extracted_subject),
        ]
        
        resolved = message
        for pattern, replacement in insertion_patterns:
            new_resolved = re.sub(pattern, replacement, resolved, flags=re.IGNORECASE)
            if new_resolved != resolved:
                logger.info(f"✅ Resolved elliptical: '{resolved}' → '{new_resolved}'")
                resolved = new_resolved
                break
        
        return resolved
    
    def _resolve_answer_to_question(
        self,
        message: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Resolve short answers to AI questions by extracting context from the question.
        
        E.g., 
        AI: "Could you please specify a region"
        User: "France"
        Should resolve to: "second best cheese cake in France"
        
        The key is to find what the AI was asking about in the conversation history.
        """
        import re
        
        if not conversation_history:
            logger.info("⚠️ No conversation history for answer resolution")
            return message
        
        # Get the last assistant message (the question)
        last_assistant_msg = None
        for msg in reversed(conversation_history):
            if msg.get('role') == 'assistant':
                last_assistant_msg = msg.get('content', '')
                break
        
        if not last_assistant_msg:
            logger.info("⚠️ No previous assistant message found")
            return message
        
        logger.info(f"🔍 Analyzing AI question: '{last_assistant_msg[:100]}...'")
        
        # Extract what the AI was asking about
        # Common patterns:
        # - "Could you please specify a [THING]" → extract THING
        # - "Which [THING] would you like?" → extract THING
        # - "What [THING] are you interested in?" → extract THING
        
        question_patterns = [
            r'(?:specify|provide|tell me|give me)\s+(?:a|an|the)?\s*(\w+)',
            r'(?:which|what)\s+(\w+)',
        ]
        
        asked_about = None
        for pattern in question_patterns:
            match = re.search(pattern, last_assistant_msg, re.IGNORECASE)
            if match:
                asked_about = match.group(1).strip()
                logger.info(f"✅ AI was asking about: '{asked_about}'")
                break
        
        # Now find the actual topic from earlier in the conversation
        # Look for user queries with ranking/comparison context (best, second best, etc.)
        # We want to find the most recent query that has explicit ranking keywords
        topic_context = None
        fallback_context = None
        
        for msg in reversed(conversation_history[:-1]):  # Skip the last assistant message
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                # Check for explicit ranking keywords
                has_explicit_ranking = any(keyword in content.lower() for keyword in 
                    ['second best', 'third best', 'fourth best', 'fifth best',
                     'first best', 'best', 'worst', 'largest', 'smallest'])
                
                if has_explicit_ranking:
                    topic_context = content
                    logger.info(f"✅ Found topic context with ranking: '{topic_context}'")
                    break
                elif not fallback_context and len(content.split()) > 2:
                    # Keep as fallback if no ranking context is found
                    fallback_context = content
        
        if not topic_context:
            topic_context = fallback_context
            if topic_context:
                logger.info(f"✅ Found fallback topic context: '{topic_context}'")
        
        if not topic_context:
            logger.info("⚠️ Could not find topic context")
            return message
        
        # Extract the main subject/query from the topic context
        # E.g., "what's the second best cheese cake" → extract "second best cheese cake"
        subject_patterns = [
            # Pattern with ordinal + superlative + subject (e.g., "second best cheese cake")
            r'(?:what\'?s?|which|who\'?s?)\s+(?:the\s+)?((?:(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+)?(?:best|worst|largest|smallest|biggest|most|least)\s+[\w\s]+?)(?:\s+in\s+|\s+for\s+|\s+of\s+|\?|$)',
            # Pattern for "what about X" (e.g., "what about cheese cake")
            r'(?:what about|how about)\s+(.+?)(?:\?|$)',
        ]
        
        extracted_topic = None
        for pattern in subject_patterns:
            match = re.search(pattern, topic_context, re.IGNORECASE)
            if match:
                extracted_topic = match.group(1).strip()
                logger.info(f"✅ Extracted topic: '{extracted_topic}'")
                break
        
        # If we couldn't extract a topic, the context itself might be elliptical (e.g., "What's the second best")
        # In this case, look further back for the actual subject
        if not extracted_topic:
            logger.info("⚠️ Topic context is elliptical, looking further back for subject")
            # Extract the ranking from the elliptical context
            ranking_match = re.search(r'((?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+)?(?:best|worst|largest|smallest|biggest|most|least)', topic_context, re.IGNORECASE)
            ranking = ranking_match.group(0) if ranking_match else None
            
            if ranking:
                logger.info(f"✅ Extracted ranking from elliptical context: '{ranking}'")
                # Look for the most recent message with a concrete subject
                # Prioritize "what about X" messages as they indicate topic shifts
                for msg in reversed(conversation_history[:-1]):
                    if msg.get('role') == 'user':
                        content = msg.get('content', '')
                        
                        # First check for "what about X" pattern (topic shift)
                        what_about_match = re.search(r'(?:what about|how about)\s+([\w\s]+?)(?:\?|$)', content, re.IGNORECASE)
                        if what_about_match:
                            subject = what_about_match.group(1).strip()
                            extracted_topic = f"{ranking} {subject}"
                            logger.info(f"✅ Combined ranking with 'what about' subject: '{extracted_topic}'")
                            break
                        
                        # Otherwise try to extract from ranking patterns
                        for pattern in subject_patterns:
                            match = re.search(pattern, content, re.IGNORECASE)
                            if match:
                                potential_topic = match.group(1).strip()
                                # Check if this has a concrete noun (not just "best" or "second best")
                                if len(potential_topic.split()) > 1:  # More than just "best"
                                    # Extract just the subject noun
                                    subject_match = re.search(r'(?:best|worst|largest|smallest|biggest|most|least)\s+([\w\s]+?)(?:\s+in\s+|\s+for\s+|\s+of\s+|\?|$)', potential_topic, re.IGNORECASE)
                                    if subject_match:
                                        subject = subject_match.group(1).strip()
                                        extracted_topic = f"{ranking} {subject}"
                                        logger.info(f"✅ Combined elliptical ranking with subject: '{extracted_topic}'")
                                        break
                        if extracted_topic:
                            break
        
        if not extracted_topic:
            logger.info("⚠️ Could not extract topic from context")
            return message
        
        # Construct the resolved message
        # Format: "[topic] in/for [answer]"
        if asked_about and asked_about.lower() in ['region', 'location', 'place', 'country', 'city']:
            resolved = f"{extracted_topic} in {message}"
        else:
            resolved = f"{extracted_topic} {message}"
        
        logger.info(f"✅ Resolved answer: '{message}' → '{resolved}'")
        return resolved
    
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
    
    def _check_needs_context(
        self,
        message: str,
        replacements: List[Dict[str, Any]],
        method: str
    ) -> bool:
        """
        Determine if this query needs conversation context to be understood.
        
        Returns True if:
        1. Coreferences were found and resolved (replacements exist)
        2. Query contains elliptical reference patterns (ordinals, follow-ups, etc.)
        3. Query is very short (< 5 words) - likely a follow-up
        
        Args:
            message: The original message
            replacements: List of coreference replacements made
            method: The coreference resolution method used
            
        Returns:
            bool: True if conversation context is needed
        """
        import re
        
        # 1. If coreferences were found, definitely needs context
        if replacements and len(replacements) > 0:
            logger.info(f"✅ needsContext=True: Found {len(replacements)} coreference replacement(s)")
            return True
        
        # 2. Check for elliptical reference patterns
        elliptical_patterns = [
            r'\b(first|second|third|fourth|fifth|next|last|previous|another|other|more|else|too|also)\b',
            r'\b(what about|how about|tell me about)\b',
            r'\b(it|that|this|them|those|these)\b',
            r'^(why|how|when|where|who|which)\b',  # Questions starting with interrogatives
            r'\b(the .+)\b',  # Definite articles often refer to previous context
        ]
        
        message_lower = message.lower()
        for pattern in elliptical_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                logger.info(f"✅ needsContext=True: Detected elliptical pattern '{pattern}' in message")
                return True
        
        # 3. Check for imperative commands (action verbs) - these are standalone
        # Common action verbs that indicate complete commands
        action_verbs = [
            'open', 'close', 'start', 'stop', 'launch', 'quit', 'exit',
            'search', 'find', 'look', 'show', 'display', 'hide',
            'create', 'make', 'build', 'delete', 'remove',
            'go', 'navigate', 'visit', 'browse',
            'play', 'pause', 'resume', 'skip',
            'send', 'email', 'message', 'call',
            'save', 'export', 'download', 'upload',
            'copy', 'paste', 'cut', 'move',
            'click', 'type', 'press', 'select'
        ]
        
        # Check if message starts with an action verb OR contains action verb after polite prefix
        words = message.lower().split()
        first_word = words[0] if words else ''
        
        # Check for elliptical patterns that use action verbs but still need context
        # e.g., "show me the first one", "find the second one", "open the next one"
        elliptical_with_verbs = [
            r'\b(show|find|get|give)\s+me\s+(the\s+)?(first|second|third|next|last|previous|another|other)\b',
            r'\b(open|show|display)\s+(the\s+)?(first|second|third|next|last|previous|another|other)\b',
        ]
        
        for pattern in elliptical_with_verbs:
            if re.search(pattern, message_lower, re.IGNORECASE):
                logger.info(f"✅ needsContext=True: Detected elliptical pattern with action verb: '{pattern}'")
                return True
        
        # Direct imperative: "open google"
        if first_word in action_verbs:
            # Check if it has a concrete target (not just ordinals/pronouns)
            if len(words) >= 2:
                second_word = words[1]
                # If second word is ordinal/pronoun, it needs context
                if second_word in ['the', 'it', 'that', 'this', 'first', 'second', 'third', 'next', 'last']:
                    logger.info(f"✅ needsContext=True: Action verb '{first_word}' followed by reference word '{second_word}'")
                    return True
            
            logger.info(f"❌ needsContext=False: Detected imperative command starting with '{first_word}'")
            return False
        
        # Polite imperative: "please open google", "can you open google", "I need you to open google"
        if len(words) >= 2:
            # Check if any action verb appears in the first 5 words (after polite prefix)
            for i, word in enumerate(words[:5]):
                if word in action_verbs:
                    # Check if there's a concrete target after the verb (not ordinal/pronoun)
                    if i < len(words) - 1:
                        next_word = words[i + 1]
                        # If next word is ordinal/pronoun, it needs context
                        if next_word in ['the', 'it', 'that', 'this', 'first', 'second', 'third', 'next', 'last']:
                            logger.info(f"✅ needsContext=True: Action verb '{word}' followed by reference word '{next_word}'")
                            return True
                        
                        logger.info(f"❌ needsContext=False: Detected imperative command with action verb '{word}' at position {i}")
                        return False
                    break
        
        # 4. Very short queries (< 5 words) are often follow-ups
        word_count = len(message.split())
        if word_count < 5:
            logger.info(f"✅ needsContext=True: Short query ({word_count} words)")
            return True
        
        # 5. No indicators found - query is likely standalone
        logger.info(f"❌ needsContext=False: No context indicators found")
        return False
