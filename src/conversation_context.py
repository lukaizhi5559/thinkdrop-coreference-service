"""
Conversation Context Tracker
Maintains semantic understanding of conversation state for better coreference resolution
"""

import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an entity mentioned in conversation"""
    text: str
    entity_type: str  # NOUN, PROPN, ORG, GPE, etc.
    modifiers: List[str] = field(default_factory=list)  # adjectives, determiners
    context: str = ""  # Full sentence context
    turn: int = 0  # Which turn it was mentioned
    position_in_list: Optional[int] = None  # Ordinal position (1st, 2nd, 3rd) when mentioned in a list
    is_most_recent: bool = False  # Track most recently mentioned entity for "that one" resolution
    
    def __str__(self):
        return f"{' '.join(self.modifiers)} {self.text}".strip()


@dataclass
class TopicState:
    """Represents the current topic/subject being discussed"""
    main_subject: str  # e.g., "restaurant", "pizza"
    subject_type: str  # e.g., "italian", "french"
    ranking: str = "best"  # "best", "second best", "third best", etc.
    filters: Dict[str, str] = field(default_factory=dict)  # location, price, etc.
    turn: int = 0
    
    def to_query(self) -> str:
        """Convert topic state to a natural query"""
        parts = []
        
        # Add ranking
        if self.ranking:
            parts.append(self.ranking)
        
        # Add subject type
        if self.subject_type:
            parts.append(self.subject_type)
            
        # Add main subject
        if self.main_subject:
            parts.append(self.main_subject)
        
        query = " ".join(parts)
        
        # Add filters
        if "location" in self.filters:
            query += f" in {self.filters['location']}"
        if "price" in self.filters:
            query += f" ({self.filters['price']})"
            
        return query


class ConversationContext:
    """
    Tracks conversation state for semantic coreference resolution
    Uses spaCy for entity extraction and semantic understanding
    """
    
    def __init__(self, nlp):
        self.nlp = nlp
        self.current_topic: Optional[TopicState] = None
        self.entity_history: List[Entity] = []  # Last N entities mentioned
        self.topic_history: List[TopicState] = []  # Topic progression
        self.max_entity_history = 40  # Increased to 40 to track "first one" references in long conversations
        self.max_topic_history = 5
        self.global_entity_position = 0  # Track global position across entire conversation
        
    def extract_entities(self, doc, turn: int) -> List[Entity]:
        """Extract meaningful entities from a spaCy doc"""
        entities = []
        
        # Extract named entities (but skip location fragments)
        # Also track which entities we've seen to avoid duplicates
        seen_entities = set()
        
        for ent in doc.ents:
            # Skip common location fragments that are often parts of larger names
            if ent.text in ['York', 'Manhattan', 'Brooklyn', 'Island', 'City']:
                continue
                
            # For GPE/LOC, only keep if it's a complete location (not a fragment)
            # E.g., keep "New York" but not "York" alone
            if ent.label_ in ['ORG', 'PERSON', 'PRODUCT']:
                # Split comma-separated entities ONLY for lists (e.g., "Carbone, Lupa, Sodi")
                # But NOT for single entities with commas (e.g., "New York, NY")
                entity_parts = [part.strip() for part in ent.text.split(',') if part.strip()]
                
                # Only split if we have multiple parts AND they look like a list (not address-like)
                # Check if parts are short (likely names, not addresses)
                if len(entity_parts) > 1 and all(len(part.split()) <= 2 for part in entity_parts):
                    # This looks like a list: "Carbone, Lupa, Sodi"
                    for entity_text in entity_parts:
                        if entity_text.lower() not in seen_entities:
                            self.global_entity_position += 1
                            entities.append(Entity(
                                text=entity_text,
                                entity_type=ent.label_,
                                context=doc.text,
                                turn=turn,
                                position_in_list=self.global_entity_position,
                                is_most_recent=True
                            ))
                            seen_entities.add(entity_text.lower())
                else:
                    # Single entity or address-like, keep as-is
                    if ent.text.lower() not in seen_entities:
                        self.global_entity_position += 1
                        entities.append(Entity(
                            text=ent.text,
                            entity_type=ent.label_,
                            context=doc.text,
                            turn=turn,
                            position_in_list=self.global_entity_position,
                            is_most_recent=True
                        ))
                        seen_entities.add(ent.text.lower())
            elif ent.label_ in ['GPE', 'LOC']:
                # Only add locations if they're multi-word or well-known single words
                if (len(ent.text.split()) > 1 or ent.text.lower() in ['brooklyn', 'manhattan', 'paris', 'london', 'tokyo']) and ent.text.lower() not in seen_entities:
                    self.global_entity_position += 1
                    entities.append(Entity(
                        text=ent.text,
                        entity_type=ent.label_,
                        context=doc.text,
                        turn=turn,
                        position_in_list=self.global_entity_position,
                        is_most_recent=True
                    ))
                    seen_entities.add(ent.text.lower())
        
        # Extract noun chunks (subjects) - but be very selective
        for chunk in doc.noun_chunks:
            # Skip pronouns, very short chunks, and common words
            if chunk.root.pos_ in ['NOUN', 'PROPN'] and len(chunk.text) > 2:
                # Skip if it's a location fragment or already seen
                if chunk.root.text.lower() in seen_entities:
                    continue
                if chunk.root.text in ['York', 'Manhattan', 'Brooklyn', 'Island', 'City']:
                    continue
                    
                # Extract modifiers (adjectives, determiners)
                modifiers = [token.text for token in chunk if token.pos_ in ['ADJ', 'DET', 'NUM']]
                
                # Only add if not a duplicate
                if chunk.root.text.lower() not in seen_entities:
                    # Only assign positions to proper nouns (likely entity names), not common nouns
                    position = None
                    is_recent = False
                    if chunk.root.pos_ == 'PROPN':
                        self.global_entity_position += 1
                        position = self.global_entity_position
                        is_recent = True
                    
                    entities.append(Entity(
                        text=chunk.root.text,
                        entity_type=chunk.root.pos_,
                        modifiers=modifiers,
                        context=doc.text,
                        turn=turn,
                        position_in_list=position,
                        is_most_recent=is_recent
                    ))
                    seen_entities.add(chunk.root.text.lower())
        
        return entities
    
    def extract_ranking(self, text: str) -> Optional[str]:
        """Extract ranking/ordinal from text (best, second best, etc.)"""
        # Ordinal + superlative patterns (including "the second cheapest")
        ordinal_match = re.search(
            r'\b(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(best|worst|largest|smallest|biggest|most|least|cheapest|expensive|costliest)',
            text,
            re.IGNORECASE
        )
        if ordinal_match:
            return ordinal_match.group(0).lower().replace('the ', '')
        
        # Bare ordinal (e.g., "the third" without superlative) - map to "X best"
        # Also handle common typos like "thrid" → "third"
        bare_ordinal_match = re.search(
            r'\b(?:the\s+)?(first|second|third|thrid|fourth|fifth|sixth|seventh|eighth|ninth|tenth)(?:\s+one)?$',
            text.strip(),
            re.IGNORECASE
        )
        if bare_ordinal_match:
            ordinal = bare_ordinal_match.group(1).lower()
            # Fix common typos
            if ordinal == 'thrid':
                ordinal = 'third'
            
            ordinal_to_ranking = {
                'first': 'best',
                'second': 'second best',
                'third': 'third best',
                'fourth': 'fourth best',
                'fifth': 'fifth best',
                'sixth': 'sixth best',
                'seventh': 'seventh best',
                'eighth': 'eighth best',
                'ninth': 'ninth best',
                'tenth': 'tenth best',
            }
            return ordinal_to_ranking.get(ordinal, f"{ordinal} best")
        
        # Just superlative
        superlative_match = re.search(
            r'\b(best|worst|largest|smallest|biggest|most|least|cheapest|expensive|costliest)\b',
            text,
            re.IGNORECASE
        )
        if superlative_match:
            return superlative_match.group(0).lower()
        
        return None
    
    def extract_location(self, doc) -> Optional[str]:
        """Extract location from spaCy doc"""
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC']:
                return ent.text
        
        # Check for "in X" pattern
        for i, token in enumerate(doc):
            if token.text.lower() == 'in' and i + 1 < len(doc):
                next_token = doc[i + 1]
                if next_token.pos_ == 'PROPN':
                    return next_token.text
        
        return None
    
    def detect_topic_shift(self, doc, text: str) -> bool:
        """Detect if this message represents a topic shift"""
        # Explicit topic shift patterns
        shift_patterns = [
            r'\b(what about|how about)\s+',
            r'\b(going back to|back to|return to)\s+',
            r'\b(instead|rather)\b',
        ]
        
        for pattern in shift_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def extract_subject_type(self, doc) -> Optional[str]:
        """Extract subject type (e.g., 'italian', 'french', 'mexican')"""
        # Look for adjectives that modify nouns (cuisine types, etc.)
        for chunk in doc.noun_chunks:
            for token in chunk:
                if token.pos_ == 'ADJ' and token.dep_ == 'amod':
                    # Check if it's a cuisine/type adjective
                    if any(keyword in chunk.root.text.lower() for keyword in ['restaurant', 'food', 'cuisine', 'place']):
                        return token.text.lower()
        
        # Look for proper nouns that might be cuisine types
        for token in doc:
            if token.pos_ == 'PROPN' or token.pos_ == 'ADJ':
                # Common cuisine types
                if token.text.lower() in ['italian', 'french', 'japanese', 'chinese', 'mexican', 
                                          'indian', 'thai', 'korean', 'vietnamese', 'spanish']:
                    return token.text.lower()
        
        return None
    
    def extract_main_subject(self, doc) -> Optional[str]:
        """Extract main subject (e.g., 'restaurant', 'pizza', 'cheese cake')"""
        # Look for meaningful noun chunks (full phrases, not just root)
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ in ['NOUN', 'PROPN']:
                # Skip very generic words, filler phrases, and ordinal numbers used as subjects
                if chunk.root.text.lower() not in ['one', 'thing', 'option', 'choice', 'it', 'that', 'this', 'info', 'information', 'options', 'more', 
                                                     'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']:
                    # Get the full noun phrase, excluding determiners at the start
                    text = chunk.text
                    if text.lower().startswith(('the ', 'a ', 'an ')):
                        text = ' '.join(text.split()[1:])
                    
                    # Skip common filler phrases
                    if text.lower() in ['more options', 'more info', 'more information']:
                        continue
                    
                    # Return the full phrase if it's meaningful
                    if text and len(text) > 1:
                        return text.lower()
        
        # Fallback: look for any NOUN, PROPN, or subject tokens that aren't in noun chunks
        # This handles cases like "What's the best pizza" where "pizza" isn't in a noun chunk
        # Also handles cases where spaCy mistagged a noun as ADJ (e.g., "pretzel" in "second best pretzel")
        for token in doc:
            # Look for NOUN/PROPN or tokens that are subjects (nsubj, nsubjpass)
            is_subject = token.dep_ in ['nsubj', 'nsubjpass', 'attr']
            if token.pos_ in ['NOUN', 'PROPN'] or (is_subject and token.pos_ == 'ADJ'):
                if token.text.lower() not in ['one', 'thing', 'option', 'choice', 'it', 'that', 'this', 'what', 'which', 'best', 'worst', 'options', 'more', 'info', 'information',
                                               'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']:
                    # Check if this is part of a compound noun (e.g., "cheese cake")
                    compound_parts = [token.text]
                    # Look backwards for compound modifiers
                    for child in token.children:
                        if child.dep_ == 'compound' and child.i < token.i:
                            compound_parts.insert(0, child.text)
                    
                    result = ' '.join(compound_parts).lower()
                    if result and len(result) > 1 and result not in ['more options', 'more info']:
                        return result
        
        return None
    
    def update_from_user_message(self, message: str, turn: int) -> None:
        """Update context from a user message"""
        logger.info(f"📝 Updating context from user message (turn {turn}): '{message}'")
        
        # Skip filler messages that don't contain meaningful content
        filler_patterns = [
            r'^(thanks|thank you|ok|okay|yes|no|sure|great|cool|nice|good|alright)(\s+(for|the|info|information|that))*$',
            r'^(i need|i want|show me|give me)\s+(more|another)\s+(options?|info|information)$',
        ]
        if any(re.match(pattern, message.strip(), re.IGNORECASE) for pattern in filler_patterns):
            logger.info(f"   ⏭️ Skipping filler message: '{message}'")
            return
        
        doc = self.nlp(message)
        
        # Extract entities and add to history
        entities = self.extract_entities(doc, turn)
        
        # Mark all existing entities as not most recent
        for entity in self.entity_history:
            entity.is_most_recent = False
        
        # Add new entities (they're already marked as most_recent=True)
        self.entity_history.extend(entities)
        self.entity_history = self.entity_history[-self.max_entity_history:]  # Keep last N
        
        # Check for topic shift
        is_topic_shift = self.detect_topic_shift(doc, message)
        
        # Extract components
        ranking = self.extract_ranking(message)
        location = self.extract_location(doc)
        subject_type = self.extract_subject_type(doc)
        main_subject = self.extract_main_subject(doc)
        
        logger.info(f"   Extracted - ranking: '{ranking}', location: '{location}', subject_type: '{subject_type}', main_subject: '{main_subject}'")
        
        # Update topic state
        if is_topic_shift or not self.current_topic:
            # New topic or explicit shift
            if main_subject or subject_type or ranking:
                # Create new topic with whatever we have
                new_topic = TopicState(
                    main_subject=main_subject or (self.current_topic.main_subject if self.current_topic else ""),
                    subject_type=subject_type or (self.current_topic.subject_type if self.current_topic else ""),
                    ranking=ranking or (self.current_topic.ranking if self.current_topic else "best"),
                    turn=turn
                )
                
                if location:
                    new_topic.filters["location"] = location
                
                # Save old topic to history
                if self.current_topic:
                    self.topic_history.append(self.current_topic)
                    self.topic_history = self.topic_history[-self.max_topic_history:]
                
                self.current_topic = new_topic
                logger.info(f"📍 New topic: {self.current_topic.to_query()}")
        else:
            # Update existing topic
            if self.current_topic:
                if ranking:
                    self.current_topic.ranking = ranking
                if location:
                    self.current_topic.filters["location"] = location
                if subject_type:
                    self.current_topic.subject_type = subject_type
                if main_subject:
                    self.current_topic.main_subject = main_subject
                
                logger.info(f"🔄 Updated topic: {self.current_topic.to_query()}")
    
    def update_from_assistant_message(self, message: str, turn: int) -> None:
        """Extract entities from assistant messages (for pronoun resolution)"""
        doc = self.nlp(message)
        
        # Extract entities from AI response
        entities = self.extract_entities(doc, turn)
        
        # Mark all existing entities as not most recent
        for entity in self.entity_history:
            entity.is_most_recent = False
        
        # Add new entities from AI message
        self.entity_history.extend(entities)
        self.entity_history = self.entity_history[-self.max_entity_history:]
        
        logger.info(f"📥 Extracted {len(entities)} entities from assistant message (turn {turn})")
    
    def resolve_elliptical(self, message: str) -> Optional[str]:
        """Resolve elliptical references using current context"""
        if not self.current_topic:
            logger.info(f"⚠️ No current topic set, cannot resolve elliptical: '{message}'")
            return None
        
        logger.info(f"🔍 Resolving elliptical with current topic: {self.current_topic.to_query()}")
        doc = self.nlp(message)
        
        # FIRST: Check for "And X" continuation patterns (e.g., "And Paris?", "And what about Tokyo?")
        and_continuation = re.match(r'^and\s+(.+?)(?:\?|$)', message.strip(), re.IGNORECASE)
        if and_continuation:
            continuation_text = and_continuation.group(1).strip()
            
            # Check if it's a simple entity continuation (e.g., "And Paris?")
            simple_entity = re.match(r'^([A-Z]\w+)$', continuation_text)
            if simple_entity:
                # This is likely a location or entity continuation
                entity_name = simple_entity.group(1)
                resolved = f"{self.current_topic.main_subject} {entity_name}"
                logger.info(f"✅ Resolved 'And X' continuation: '{message}' → '{resolved}'")
                return resolved
            
            # Check if it's "And what about X"
            and_what_about = re.match(r'^what about\s+(.+)$', continuation_text, re.IGNORECASE)
            if and_what_about:
                new_subject = and_what_about.group(1).strip()
                # Apply same logic as regular "what about" but preserve ranking
                resolved = f"{self.current_topic.main_subject} {new_subject}"
                logger.info(f"✅ Resolved 'And what about X': '{message}' → '{resolved}'")
                return resolved
        
        # SECOND: Check for "another/next/cheapest option/one/choice" before general "what about X"
        option_match = re.match(r'^(?:what about|how about|show me)?\s*(?:the\s+)?(another|other|next|last|previous|cheapest)\s+(one|option|choice)$', message.strip(), re.IGNORECASE)
        if option_match:
            modifier = option_match.group(1).lower()
            # Map modifiers to rankings or keep current
            if modifier == 'cheapest':
                self.current_topic.ranking = 'cheapest'
            elif modifier in ['another', 'other', 'next']:
                # Keep current ranking, just return another instance
                pass
            
            resolved = f"what's {self.current_topic.to_query()}"
            logger.info(f"✅ Resolved '{modifier} option' (early): '{message}' → '{resolved}'")
            return resolved
        
        # SECOND: Check for bare ordinals like "how about the third" before general "what about X"
        bare_ordinal_early_match = re.match(r'^(?:how about|what about)\s+(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)(?:\s+one)?$', message.strip(), re.IGNORECASE)
        if bare_ordinal_early_match:
            ordinal = bare_ordinal_early_match.group(1).lower()
            ordinal_to_ranking = {
                'first': 'best',
                'second': 'second best',
                'third': 'third best',
                'fourth': 'fourth best',
                'fifth': 'fifth best',
                'sixth': 'sixth best',
                'seventh': 'seventh best',
                'eighth': 'eighth best',
                'ninth': 'ninth best',
                'tenth': 'tenth best',
            }
            self.current_topic.ranking = ordinal_to_ranking.get(ordinal, f"{ordinal} best")
            resolved = f"what's {self.current_topic.to_query()}"
            logger.info(f"✅ Resolved bare ordinal (early): '{message}' → '{resolved}'")
            return resolved
        
        # Check for "going back to X, what about Y" pattern
        going_back_match = re.search(r'going back to\s+(\w+),\s*what about\s+(.+?)(?:\?|$)', message, re.IGNORECASE)
        if going_back_match:
            topic_return = going_back_match.group(1).lower()
            new_subject = going_back_match.group(2).strip()
            
            # Extract location from new_subject if present
            location_doc = self.nlp(new_subject)
            location = self.extract_location(location_doc)
            
            # Build resolved query
            resolved = f"what's best {topic_return[:-1] if topic_return.endswith('s') else topic_return}"
            if location:
                resolved += f" in {location}"
            
            logger.info(f"✅ Resolved 'going back to': '{message}' → '{resolved}'")
            return resolved
        
        # Check for temporal shifts: "tomorrow", "yesterday", "next week", etc.
        temporal_match = re.search(r'\b(tomorrow|yesterday|tonight|today|next\s+\w+|last\s+\w+|this\s+\w+)\b', message, re.IGNORECASE)
        if temporal_match:
            temporal_modifier = temporal_match.group(1).lower()
            
            # Check if this is a simple temporal shift (e.g., "what about tomorrow")
            simple_temporal = re.match(r'^(what about|how about)\s+(tomorrow|yesterday|tonight|today|next\s+\w+|last\s+\w+|this\s+\w+)$', message.strip(), re.IGNORECASE)
            if simple_temporal:
                # Preserve current topic and add temporal modifier
                resolved = f"{self.current_topic.main_subject} {temporal_modifier}"
                logger.info(f"✅ Resolved temporal shift: '{message}' → '{resolved}'")
                return resolved
        
        # Check for "what about X" topic shift pattern
        what_about_match = re.search(r'\b(what about|how about)\s+(.+?)(?:\?|$)', message, re.IGNORECASE)
        if what_about_match:
            new_subject_text = what_about_match.group(2).strip()
            
            # Check if it's a temporal modifier (e.g., "what about tomorrow")
            if re.match(r'^(tomorrow|yesterday|tonight|today|next\s+\w+|last\s+\w+|this\s+\w+)$', new_subject_text, re.IGNORECASE):
                # Preserve current topic and add temporal modifier
                resolved = f"{self.current_topic.main_subject} {new_subject_text}"
                logger.info(f"✅ Resolved temporal 'what about': '{message}' → '{resolved}'")
                return resolved
            
            # Check if it's just a location (e.g., "what about in Los Angeles")
            location_only_match = re.match(r'^in\s+(.+)$', new_subject_text, re.IGNORECASE)
            if location_only_match:
                # This is just a location refinement, not a topic shift
                location = location_only_match.group(1).strip()
                self.current_topic.filters["location"] = location
                resolved = self.current_topic.to_query()
                logger.info(f"✅ Resolved location-only 'what about': '{message}' → '{resolved}'")
                return resolved
            
            # Check if it contains a ranking + modifier (e.g., "the best budget option")
            ranking_modifier_match = re.match(
                r'^(the\s+)?(best|worst|cheapest|expensive)\s+(\w+)\s+(option|one|choice)$',
                new_subject_text,
                re.IGNORECASE
            )
            if ranking_modifier_match:
                ranking = ranking_modifier_match.group(2).lower()
                modifier = ranking_modifier_match.group(3).lower()
                # Apply ranking and modifier to current topic
                self.current_topic.ranking = ranking
                resolved = f"what's {ranking} {modifier} {self.current_topic.subject_type or ''} {self.current_topic.main_subject}".strip()
                logger.info(f"✅ Resolved ranking + modifier: '{message}' → '{resolved}'")
                return resolved
            
            # Check if it's an elliptical pattern (e.g., "the second best")
            is_elliptical_subject = re.match(
                r'^(the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(best|worst|largest|smallest|biggest|most|least)$',
                new_subject_text,
                re.IGNORECASE
            )
            
            if not is_elliptical_subject:
                # Check if current topic is a question pattern (e.g., "what time", "what color", "how much")
                is_question_pattern = self.current_topic.main_subject and any(
                    self.current_topic.main_subject.startswith(q) 
                    for q in ['what ', 'how ', 'when ', 'where ', 'why ', 'which ']
                )
                
                # Check if new subject is just a location (proper noun)
                new_subject_doc = self.nlp(new_subject_text)
                is_location_shift = any(ent.label_ == 'GPE' for ent in new_subject_doc.ents)
                
                if is_question_pattern and is_location_shift:
                    # Preserve the question pattern and just update location
                    # e.g., "What time is it in France" + "what about brazil" → "What time is it in Brazil"
                    location = new_subject_text
                    self.current_topic.filters["location"] = location
                    
                    # Reconstruct the question with new location
                    resolved = f"{self.current_topic.main_subject}"
                    if location:
                        resolved += f" in {location}"
                    
                    logger.info(f"✅ Resolved question + location shift: '{message}' → '{resolved}'")
                    return resolved
                
                # This is a topic shift - extract new subject type and/or main subject
                subject_type = self.extract_subject_type(doc)
                main_subject = self.extract_main_subject(doc)
                
                logger.info(f"🔍 Topic shift detected - subject_type: '{subject_type}', main_subject: '{main_subject}'")
                
                # If we have either a subject type or main subject, this is a valid topic shift
                if subject_type or main_subject:
                    # Apply current ranking to new subject
                    resolved = f"what's {self.current_topic.ranking}"
                    if subject_type:
                        resolved += f" {subject_type}"
                    if main_subject and main_subject != subject_type:
                        # Only add main subject if it's different from subject type
                        resolved += f" {main_subject}"
                    elif main_subject and not subject_type:
                        # No subject type, just use main subject
                        resolved += f" {main_subject}"
                    
                    logger.info(f"✅ Resolved topic shift: '{message}' → '{resolved}'")
                    return resolved
                else:
                    # Fallback: if we can't extract anything, just use the text after "what about"
                    resolved = f"what's {self.current_topic.ranking} {new_subject_text}"
                    logger.info(f"✅ Resolved topic shift (fallback): '{message}' → '{resolved}'")
                    return resolved
        
        # Check for location refinement patterns first (e.g., "in Brooklyn though", "in San Francisco though")
        location_refinement_match = re.match(r'^in\s+(.+?)(?:\s+though)?$', message.strip(), re.IGNORECASE)
        if location_refinement_match:
            location = location_refinement_match.group(1).strip()
            self.current_topic.filters["location"] = location
            resolved = self.current_topic.to_query()
            logger.info(f"✅ Resolved location refinement: '{message}' → '{resolved}'")
            return resolved
        
        # Check for other elliptical patterns
        elliptical_patterns = [
            r'\b(what\'?s?|which|who\'?s?)\s+(the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(best|worst|largest|smallest|biggest|most|least)',
            r'\b(the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(best|worst|largest|smallest|biggest|most|least|cheapest)',
            r'\b(the\s+)?(next|last|previous|another|other)\s+(one|option|choice)',
            r'\b(what about|how about)\s+(the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(best|worst|largest|smallest|biggest|most|least)',
            r'\b(what about|how about)\s+(the\s+)?(next|last|previous|another|other|cheapest)\s+(one|option|choice)',  # "what about the next option"
            r'\b(how about|what about)\s+(.+?)\s+(specifically|though|again)',  # "how about pizza specifically"
            r'^(the\s+)?(third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)$',  # Just "the third"
            r'^(the\s+)?(second|third)\s+(one)$',  # "the second one"
            r'\b(the\s+)?(best|worst)\s+(one)\s+(there|here)',  # "the best one there"
            r'^in\s+.+?(?:\s+though)?$',  # "in Brooklyn though" or "in San Francisco though"
            r'\bshow\s+me\s+(another|other)\s+(one|option)',  # "show me another one"
            r'going back to\s+\w+',  # "going back to hotels"
        ]
        
        is_elliptical = any(re.search(pattern, message, re.IGNORECASE) for pattern in elliptical_patterns)
        
        if not is_elliptical:
            return None
        
        # Special handling for "another option/one/choice"
        another_match = re.match(r'^(?:what about|how about)\s+(another|other)\s+(one|option|choice)$', message.strip(), re.IGNORECASE)
        if another_match:
            # "another" typically means maintaining current ranking
            resolved = self.current_topic.to_query()
            resolved = f"what's {resolved}"
            logger.info(f"✅ Resolved 'another option': '{message}' → '{resolved}'")
            return resolved
        
        # Special handling for "the best/worst one there/here"
        best_one_there_match = re.match(r'^(?:the\s+)?(best|worst|cheapest)\s+one\s+(there|here)$', message.strip(), re.IGNORECASE)
        if best_one_there_match:
            ranking = best_one_there_match.group(1).lower()
            self.current_topic.ranking = ranking
            resolved = self.current_topic.to_query()
            logger.info(f"✅ Resolved 'the {ranking} one there': '{message}' → '{resolved}'")
            return resolved
        
        # Special handling for bare ordinals like "the third" or "the second one"
        # Also handles "how about the third"
        bare_ordinal_match = re.match(r'^(?:how about|what about)?\s*(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)(?:\s+one)?$', message.strip(), re.IGNORECASE)
        if bare_ordinal_match:
            ordinal = bare_ordinal_match.group(1).lower()
            # Map ordinal to ranking
            ordinal_to_ranking = {
                'first': 'best',
                'second': 'second best',
                'third': 'third best',
                'fourth': 'fourth best',
                'fifth': 'fifth best',
                'sixth': 'sixth best',
                'seventh': 'seventh best',
                'eighth': 'eighth best',
                'ninth': 'ninth best',
                'tenth': 'tenth best',
            }
            self.current_topic.ranking = ordinal_to_ranking.get(ordinal, f"{ordinal} best")
            
            # Build resolved query with updated ranking
            resolved = self.current_topic.to_query()
            
            # Add question word if present
            if re.match(r'\b(how about|what about)\b', message, re.IGNORECASE):
                resolved = f"what's {resolved}"
            
            logger.info(f"✅ Resolved bare ordinal: '{message}' → '{resolved}'")
            return resolved
        
        # Extract any new ranking from the message
        new_ranking = self.extract_ranking(message)
        if new_ranking:
            self.current_topic.ranking = new_ranking
        
        # Build resolved query
        resolved = self.current_topic.to_query()
        
        # Check if message starts with question word
        if re.match(r'\b(what\'?s?|which|who\'?s?|how)\b', message, re.IGNORECASE):
            # Keep the question structure
            question_word = re.match(r'\b(what\'?s?|which|who\'?s?|how)\b', message, re.IGNORECASE).group(0)
            resolved = f"{question_word} {resolved}"
        
        logger.info(f"✅ Resolved elliptical: '{message}' → '{resolved}'")
        return resolved
    
    def resolve_pronoun(self, message: str, doc) -> Optional[str]:
        """Resolve pronouns using entity history"""
        # Check for pronouns and demonstratives
        pronouns = ['it', 'that', 'this', 'them', 'those', 'these']
        has_pronoun = any(token.text.lower() in pronouns for token in doc)
        
        # Special handling for "that one", "this one", "the first one"
        message_lower = message.lower()
        that_one_match = re.search(r'\b(that|this)\s+one\b', message_lower)
        ordinal_match = re.search(r'\bthe\s+(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+one\b', message_lower)
        
        if not (has_pronoun or that_one_match or ordinal_match) or not self.entity_history:
            return None
        
        resolved = message  # Start with original message
        
        # Handle ordinal references FIRST: "the first one", "the second one"
        if ordinal_match:
            ordinal_text = ordinal_match.group(1)
            ordinal_map = {
                'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
                'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10
            }
            target_position = ordinal_map.get(ordinal_text)
            
            logger.info(f"🔍 Looking for entity at position {target_position}")
            logger.info(f"📋 Entity history: {[(e.text, e.position_in_list) for e in self.entity_history]}")
            
            if target_position:
                # Find entity at that position
                for entity in self.entity_history:
                    if entity.position_in_list == target_position:
                        resolved = re.sub(r'\bthe\s+(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+one\b', 
                                        entity.text, resolved, flags=re.IGNORECASE, count=1)
                        logger.info(f"✅ Resolved ordinal reference: 'the {ordinal_text} one' → '{entity.text}'")
                        break
                else:
                    logger.info(f"⚠️ No entity found at position {target_position}")
        
        # Handle "that one" / "this one" - use most recent entity
        if that_one_match and resolved == message:  # Only if not already resolved
            for entity in reversed(self.entity_history):
                if entity.is_most_recent:
                    resolved = re.sub(r'\b(that|this)\s+one\b', entity.text, resolved, flags=re.IGNORECASE, count=1)
                    logger.info(f"✅ Resolved 'that/this one' → '{entity.text}'")
                    break
        
        # Handle regular pronouns (it, that, this, etc.)
        if has_pronoun:
            # Get the most recent concrete entity
            # NOW INCLUDES PERSON entities for "it" (users often say "it" for people)
            last_entity = None
            for entity in reversed(self.entity_history):
                # Prioritize PERSON, ORG, PRODUCT entities
                if entity.entity_type in ['PERSON', 'ORG', 'PRODUCT']:
                    last_entity = entity
                    break
                elif entity.entity_type == 'PROPN':
                    # For PROPN, prefer multi-word entities or capitalized words
                    if len(entity.text.split()) > 1 or (entity.text[0].isupper() and len(entity.text) > 3):
                        last_entity = entity
                        break
            
            if not last_entity:
                # Fallback: use any entity
                for entity in reversed(self.entity_history):
                    if entity.entity_type in ['ORG', 'PROPN', 'PRODUCT', 'PERSON']:
                        last_entity = entity
                        break
            
            if last_entity:
                # Replace pronoun with entity
                for pronoun in pronouns:
                    pattern = r'\b' + pronoun + r'\b'
                    new_resolved = re.sub(pattern, last_entity.text, resolved, flags=re.IGNORECASE, count=1)
                    if new_resolved != resolved:
                        logger.info(f"✅ Resolved pronoun '{pronoun}' → '{last_entity.text}'")
                        resolved = new_resolved
                        break
        
        # Return resolved message if anything changed
        if resolved != message:
            logger.info(f"✅ Final resolved: '{message}' → '{resolved}'")
            return resolved
        
        return None
    
    def resolve_short_answer(self, message: str, last_assistant_message: str) -> Optional[str]:
        """Resolve short answers to AI questions"""
        # Check if message is short (1-3 words)
        if len(message.split()) > 3:
            return None
        
        if not self.current_topic:
            return None
        
        # Check if assistant asked a question
        is_question = '?' in last_assistant_message or any(
            phrase in last_assistant_message.lower() 
            for phrase in ['could you', 'would you', 'please specify', 'which', 'what', 'where']
        )
        
        if not is_question:
            return None
        
        # Determine what was asked about
        doc = self.nlp(last_assistant_message)
        location = self.extract_location(doc)
        
        # Check if asking about location
        if any(word in last_assistant_message.lower() for word in ['region', 'location', 'place', 'area', 'neighborhood', 'city', 'where']):
            # Answer is a location
            self.current_topic.filters["location"] = message
            resolved = f"{self.current_topic.to_query()}"
            logger.info(f"✅ Resolved short answer (location): '{message}' → '{resolved}'")
            return resolved
        
        # Check if asking about price/budget
        if any(word in last_assistant_message.lower() for word in ['budget', 'price', 'cost', 'expensive', 'cheap']):
            self.current_topic.filters["price"] = message
            resolved = f"{self.current_topic.to_query()}"
            logger.info(f"✅ Resolved short answer (price): '{message}' → '{resolved}'")
            return resolved
        
        # Check if asking about style/preference (casual, formal, etc.)
        if any(word in last_assistant_message.lower() for word in ['casual', 'formal', 'fine dining', 'style', 'preference', 'looking for']):
            # Answer is a style preference - append to current topic
            resolved = f"{message} {self.current_topic.to_query()}"
            logger.info(f"✅ Resolved short answer (style): '{message}' → '{resolved}'")
            return resolved
        
        return None
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current context for debugging"""
        return {
            "current_topic": self.current_topic.to_query() if self.current_topic else None,
            "entity_count": len(self.entity_history),
            "last_entities": [str(e) for e in self.entity_history[-3:]],
            "topic_history": [t.to_query() for t in self.topic_history]
        }
