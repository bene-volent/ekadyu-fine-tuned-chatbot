import logging
import re
import time
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread
from accelerate import init_empty_weights
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FineTunedChatbot:
    def __init__(self, model_path="deepseek-1.5B-finetuned"):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()]
        )
        
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        
        try:
            # Model Configuration
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            # Free GPU memory
            torch.cuda.empty_cache()
            
            # Load model and tokenizer
            self.model, self.tokenizer = self._load_model_and_tokenizer()
            self.logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize chatbot: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise RuntimeError(f"Chatbot initialization failed: {str(e)}")
        
    def _load_model_and_tokenizer(self):
        try:
            self.logger.info(f"Loading tokenizer from {self.model_path}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
            
            self.logger.info(f"Loading model from {self.model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map='balanced',
                quantization_config=self.quantization_config
            )
            
            return model, tokenizer
        except Exception as e:
            self.logger.error(f"Failed to load model or tokenizer: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _is_question_in_context(self, question, context):
        """Determine if a question is relevant to the provided context"""
        try:
            if not context:
                return True
                
            documents = [context, question]
            vectorizer = TfidfVectorizer().fit_transform(documents)
            vectors = vectorizer.toarray()
            similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
            return similarity > 0.2  # Adjust threshold as needed
        except Exception as e:
            self.logger.warning(f"Error checking question context: {str(e)}")
            # Default to True on error to avoid blocking valid queries
            return True
    
    def _classify_question_type(self, question):
        """Classify the type of question for appropriate handling"""
        try:
            question_lower = question.lower()
            explanatory_keywords = ["what is", "who is", "define", "explain", "elaborate", "clarify"]
            detailed_keywords = ["why", "how", "describe", "compare", "analyze", "detail"]
            direct_keywords = ["is", "can", "does", "did", "has", "have", "will", "are", "was"]
            bullet_point_keywords = ["list", "bullet points"]
            powerpoint_keywords = ["powerpoint", "slides", 'presentation', 'slide']

            # Check most specific keyword types first
            if any(word in question_lower for word in powerpoint_keywords):
                return "powerpoint"
            elif any(word in question_lower for word in bullet_point_keywords):
                return "bullet_points"
            elif any(word in question_lower for word in detailed_keywords):
                return "detailed"
            elif any(word in question_lower for word in explanatory_keywords):
                return "explanatory"
            elif any(word in question_lower for word in direct_keywords):
                return "direct"
            else:
                return "general"
        except Exception as e:
            self.logger.warning(f"Error classifying question: {str(e)}")
            # Default to general on error
            return "general"
    
    def _determine_max_word_count(self, question_type):
        """Determine appropriate word count limit based on question type"""
        word_count_mapping = {
            "direct": 50,
            "explanatory": 200,
            "detailed": 250,
            "general": 100,
            "bullet_points": 700,
            "powerpoint": 4000
        }
        return word_count_mapping.get(question_type, 100)
    
    def _generate_prompt(self, question, question_type, max_word_count):
        """
        Generates a prompt for an educational AI model based on the question type.
        """
        try:
            prompt_templates = {
                "direct": (
                    "<think>\n"
                    "You are Ekadyu Chatbot, an AI designed to provide educational answers.\n"
                    "Provide a concise, factual answer in under {max_word_count} words.\n"
                    
                    "Focus on educational content only. No greetings or fluff.\n\n"
                    "Question: {question}\n"
                    "Answer:\n"
                    "</think>"
                ),
                "explanatory": (
                    "<think>\n"
                    "You are Ekadyu Chatbot, an AI designed to provide educational answers.\n"
                    "Give a detailed explanation with examples in under {max_word_count} words.\n"
                    "Include definitions, meanings, and examples.\n"
                    
                    "Educational content only. No conversational elements.\n\n"
                    "Question: {question}\n"
                    "Explanation:\n"
                    "</think>"
                ),
                "detailed": (
                    "<think>\n"
                    "You are Ekadyu Chatbot, an AI designed to provide educational answers.\n"
                    "Provide a structured, step-by-step explanation with a conclusion in under {max_word_count} words.\n"
                    "Structure: Definition, breakdown, examples, conclusion.\n"
                    "Use clear headings and paragraphs. Educational content only.\n"
                    "For mathematical expressions, use LaTeX notation: inline math must be enclosed within $...$ and block math must be enclosed with $$...$$\n\n"
                    "Question: {question}\n"
                    "Explanation:\n"
                    "</think>"
                ),
                "general": (
                    "<think>\n"
                    "You are Ekadyu Chatbot, an AI designed to provide educational answers.\n"
                    "Give a thoughtful, contextual answer in under {max_word_count} words.\n"
                    "Include basic background info and examples. Educational content only.\n"
                    "For mathematical expressions, use LaTeX notation: inline math must be enclosed within $...$ and block math must be enclosed with $$...$$\n\n"
                    "Question: {question}\n"
                    "Answer:\n"
                    "</think>"
                ),
                "bullet_points": (
                    "<think>\n"
                    "You are Ekadyu Chatbot, an AI designed to provide educational answers.\n"
                    "Provide key information as comprehensive bullet points in under {max_word_count} words.\n"
                    "Format each bullet point with * at the beginning.\n"
                    "Each bullet point should be standalone and include necessary context.\n"
                    
                    "Educational content only. No conversational elements.\n\n"
                    "Topic: {question}\n"
                    "Bullet Points:\n"
                    "</think>"
                ),
                "powerpoint": (
                    "<think>\n"
                    "You are Ekadyu Chatbot, an AI designed to provide educational answers.\n"
                    "Create a complete PowerPoint presentation with 5-9 content-rich slides in under {max_word_count} words.\n"
                    
                    "Each slide should be formatted as:\n"
                    "<slide>\n"
                    "# Slide Title\n"
                    "* Bullet point 1\n"
                    "* Bullet point 2\n"
                    "Additional explanatory paragraph if needed.\n"
                    "</slide>\n\n"
                    "Include a title slide at the beginning and conclusion slide at the end.\n"
                    "Topic: {question}\n"
                    "Slides:\n"
                    "</think>"
                )
            }

            template = prompt_templates.get(question_type, prompt_templates["general"])
            return template.format(question=question, max_word_count=max_word_count)
        except Exception as e:
            self.logger.error(f"Error generating prompt: {str(e)}")
            # Fallback to a simple prompt
            return f"<think>\nAnswer the following question: {question}\n</think>"
    
    def _get_reply(self, response):
        """Extract the actual reply by removing <think> tags and their content"""
        try:
            cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            return cleaned
        except Exception as e:
            self.logger.warning(f"Error extracting reply: {str(e)}")
            # Return original if processing fails
            return response
    
    def _is_response_complete(self, response):
        """Check if a response appears to be complete based on ending punctuation"""
        try:
            cleaned = re.sub(r"<[^>]*>", "", response).strip()
            return cleaned.endswith(('.', '!', '?', '"', '\'')) or len(cleaned.split()) < 30
        except Exception as e:
            self.logger.warning(f"Error checking response completeness: {str(e)}")
            # Assume incomplete on error
            return False
    
    def _preprocess_bullet_points(self, content):
        """Lightly preprocess bullet points for better display"""
        try:
            # Remove any title/header lines (non-bullet points at the start)
            lines = content.split('\n')
            result_lines = []
            
            for line in lines:
                # Make sure each bullet point starts with * if it doesn't already
                stripped = line.strip()
                if stripped and not stripped.startswith(('*', '-', '•')):
                    # Skip header lines or convert to bullets if it looks like it should be one
                    if len(result_lines) > 0 and not any(bullet in line for bullet in [':', '.', '?']):
                        result_lines.append(f"* {stripped}")
                else:
                    result_lines.append(line)
            
            return '\n'.join(result_lines)
        except Exception as e:
            self.logger.warning(f"Error preprocessing bullet points: {str(e)}")
            return content
    
    def ask(self, question, context=None):
        """
        Ask a question to the chatbot and get a complete response.
        
        Args:
            question: The question to answer
            context: Optional context for the question
            
        Returns:
            Dictionary with answer and completion status
        """
        try:
            if context and not self._is_question_in_context(question, context):
                return {'answer': "Cannot answer this question as it is out of scope.", 'complete': True}

            question_type = self._classify_question_type(question)

            if question_type == "bullet_points" or question_type == "powerpoint":
                return {'answer': "Please use the specialized endpoints for bullet points or presentations.", 'complete': False}

            max_word_count = self._determine_max_word_count(question_type)
            prompt = self._generate_prompt(question, question_type, max_word_count)

            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            input_token_size = inputs["input_ids"].shape[1]

            max_new_tokens = round((max_word_count) * 1.3 + input_token_size)

            outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,      # Controlled response length [1][3]
                        do_sample=True,
                        temperature=0.3,         # Balanced determinism [2][5]
                        top_p=0.85,              # Focused token selection [3][5]
                        top_k=40,                # Enhanced coherence [3]
                        repetition_penalty=1.8,  # Strong anti-repetition [1][3]
                        frequency_penalty=1.5,   # Term variation [3][4]
                        presence_penalty=1.2,    # Controlled novelty [1][4]
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            cleaned_response = self._get_reply(response)
            is_complete = self._is_response_complete(response)
            
            return {'answer': cleaned_response, 'complete': is_complete}
        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {'answer': f"I encountered an error while processing your question: {str(e)}", 'complete': False}

    def ask_stream(self, question, context=None):
        """
        Stream the response token by token as they're generated by the model.
        
        Args:
            question: The question to answer
            context: Optional context for the question
            
        Yields:
            Tokens as they're generated
        """
        try:
            if context and not self._is_question_in_context(question, context):
                yield "Cannot answer this question as it is out of scope."
                return

            # Classify and prepare prompt
            question_type = self._classify_question_type(question)

            if question_type == "bullet_points" or question_type == "powerpoint":
                yield "Please use the specialized endpoints for bullet points or presentations."
                return

            max_word_count = self._determine_max_word_count(question_type)
            prompt = self._generate_prompt(question, question_type, max_word_count)

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            input_token_size = inputs["input_ids"].shape[1]
            max_new_tokens = round((max_word_count) * 1.3 + input_token_size)

            # Create a streamer
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
            
            # Set up generation arguments
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "pad_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer,
            }

            # Start generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Track the full response
            full_response = ""
            saw_think_close = False
            
            # Iterate through the streamed tokens
            for token in streamer:
                # Add the token to the current response
                full_response += token
                
                # Process think tags for the streaming output
                if "</think>" in full_response and not saw_think_close:
                    # Once we see the closing tag, we only want to show content after it
                    saw_think_close = True
                    # Extract only what's after the last </think> tag
                    content_parts = full_response.split("</think>")
                    if len(content_parts) > 1:
                        clean_response = content_parts[-1].strip()
                        # Yield the complete clean response so far
                        yield clean_response
                elif saw_think_close:
                    # We're past the think tags, yield the full clean response each time
                    content_parts = full_response.split("</think>")
                    if len(content_parts) > 1:
                        clean_response = content_parts[-1].strip()
                        yield clean_response
            
            # If we never saw the closing think tag, clean the response
            if not saw_think_close and full_response:
                clean_response = self._get_reply(full_response)
                yield clean_response
                
        except Exception as e:
            self.logger.error(f"Error in streaming response: {str(e)}")
            self.logger.error(traceback.format_exc())
            yield f"I encountered an error while generating your response: {str(e)}"



    def get_conversational_response(self, question,context=None):
        """
        Get a conversational response to a user message.
        
        Args:
            question: The user message
            context: Optional context for the message
            
        Returns:
            Dictionary with answer and completion status
        """
        try:
            if question.strip() == "":
                return {'answer': "Please provide a valid message.", 'complete': True}

            # Create conversational prompt - keeping it simple and friendly
            prompt = (
                "<think>\n"
                "You are Ekadyu Chatbot having a casual conversation with a user.\n"
                "Keep your response friendly, brief, and conversational. Don't use markdown or any notation\n"
                "Respond naturally as if chatting with a friend, not as a Q&A system.\n"
                "Limit to 2-3 sentences maximum. Do not give explanations unless asked.\n"
                + (f"Consider this context in your response: {context}\n" if context else "") +
                f"User message: {question}\n"
                "Response:\n"
                "</think>"
            )

            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,  # Keep responses short
                do_sample=True,
                temperature=0.7,     # More creative for conversation
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            cleaned_response = self._get_reply(response)
            
            is_complete = self._is_response_complete(response)

            if not is_complete:
                # If the response is incomplete, remove the incomplete last sentence
                sentences = re.split(r'(?<=[.!?]) +', cleaned_response)
                sentences = [s for s in sentences if s.strip()]  # Remove empty sentences
                if len(sentences) > 1:
                    cleaned_response = ' '.join(sentences[:-1]).strip()
                else:
                    cleaned_response = sentences[0].strip()
               
            return {'answer': cleaned_response, 'complete': self._is_response_complete(response)}
        except Exception as e:
            self.logger.error(f"Error in conversational response: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {'answer': "Sorry, I had trouble processing your message.", 'complete': False}
        

    def _generate_powerpoint_prompt(self, topic, max_word_count):
        """
        Generate a specialized prompt for PowerPoint presentations with detailed formatting instructions,
        without adding complex HTML tags.
        """
        return (
            "<think>\n"
                    "You are Ekadyu Chatbot, an AI designed to provide educational answers.\n"
            f"Create a professional final PowerPoint presentation with 5-9 slides in under {max_word_count} words.\n\n"
            "FORMAT REQUIREMENTS:\n"
            "1. First slide must be a title slide with presentation title and subtitle\n"
            "2. Final slide must be a conclusion/summary slide\n"
            "3. Each slide must have: Title format\n"
            "4. Use consistent formatting throughout\n"
            "5. Include these elements as needed:\n"
            "   - Bullet points: * Main point\n"
            "   - Sub-bullets: → Sub-point\n"
            "   - Numbered lists: 1. Step one\n"
            "   - Bold text: **important text**\n"
            "   - Diagrams: [Diagram: diagram description]\n"
            "   - Tables: [Table: table content]\n"
            "   - Charts: [Chart: chart description]\n\n"
            "6. Do not include any conversational elements or greetings\n"
            "EXAMPLE SLIDE:\n"
            "---\n"
            "# Introduction to Machine Learning\n\n"
            "* Machine learning is a branch of artificial intelligence\n"
            "* Key applications include:\n"
            "  → Image recognition\n"
            "  → Natural language processing\n"
            "  → Predictive analytics\n\n"
            "Machine learning systems learn from data without explicit programming.\n"
            "---\n\n"
            f"Topic: {topic}\n"
            "Presentation:\n"
            "</think>"
        )


    
    # def _generate_powerpoint_prompt(self, topic, max_word_count):
    #     """
    #     Generate a specialized prompt for PowerPoint presentations with simple formatting instructions.
    #     """
    #     return (
    #         "<think>\n"
    #         f"Create a professional PowerPoint presentation with 5-7 slides in under {max_word_count} words.\n\n"
    #         "FORMAT REQUIREMENTS:\n"
    #         "1. Separate each slide with triple dashes (---)\n"
    #         "2. Start each slide with a title using # format\n"
    #         "3. First slide should be a title slide\n"
    #         "4. Last slide should be a conclusion slide\n"
    #         "5. Use bullet points (*) for key information\n"
    #         "6. Do not use any complex formatting\n\n"
    #         "EXAMPLE SLIDE FORMAT:\n"
    #         "---\n"
    #         "# Introduction to Machine Learning\n\n"
    #         "* Machine learning is a branch of artificial intelligence\n"
    #         "* It allows computers to learn from data\n"
    #         "* Used in many modern applications\n"
    #         "---\n\n"
    #         f"Topic: {topic}\n"
    #         "Presentation:\n"
    #         "</think>"
    #     )
    

    def _generate_examples_prompt(self, content, example_count=2):
        """
        Generate a prompt to request examples related to the content.
        
        Args:
            content: The content to generate examples for
            example_count: Number of examples to request
            
        Returns:
            Formatted prompt for example generation
        """
        try:
            # Extract key concepts from the content
            content_summary = content[:500] if len(content) > 500 else content
            
            prompt = (
                "<think>\n"
                "You are Ekadyu Chatbot, an AI designed to provide educational examples.\n"
                f"Based on the following content, generate {example_count} practical, real-world examples.\n"
                "Each example should:\n"
                "1. Be concrete and specific in less that 30 words\n"
                "2. Demonstrate practical application\n"
                "3. Be easy to understand\n"
                "4. Include very small explanation of why it's relevant\n\n"
                "5. No markdown or complex formatting\n\n"
                f"Content: {content_summary}\n\n"
                "Examples:\n"
                "</think>"
            )
            
            return prompt
        except Exception as e:
            self.logger.warning(f"Error generating examples prompt: {str(e)}")
            return f"<think>Generate {example_count} examples for: {content[:100]}...</think>"


    def create_presentation(self, topic, context=None,):
        """
        Generate a presentation on the given topic with minimal preprocessing.
        
        Args:
            topic: The topic for the presentation
            context: Optional context for the topic
                
        returns:
            Dictionary with answer and completion status
        """
        try:
            if context and not self._is_question_in_context(topic, context):
                return {'answer': "Cannot create a presentation on this topic as it is out of scope.", 'complete': True}
            
            question_type = "powerpoint"
            max_word_count = self._determine_max_word_count(question_type)
            
            # Use the specialized PowerPoint prompt instead of the generic one
            prompt = self._generate_powerpoint_prompt(topic, max_word_count)

            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            input_token_size = inputs["input_ids"].shape[1]

            # Increase token allocation for presentations (1.5x instead of 1.3x)
            max_new_tokens = round((max_word_count) * 1.5 + input_token_size)


            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,  # Slightly higher temperature for presentations
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            clean_response = self._get_reply(response)
            
            # Only do minimal preprocessing - keep slide tags intact for frontend formatting
            preprocessed_response = self._preprocess_presentation(clean_response)

            return {'answer': preprocessed_response, 'complete': self._is_response_complete(response)}
        
            
                
        except Exception as e:
            error_msg = f"Error generating presentation: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            return {'answer': error_msg, 'complete': False}

    def create_bullet_points(self, topic, context=None, stream=False):
        """
        Generate bullet points for the given topic.
        
        Args:
            topic: The topic to generate bullet points for
            context: Optional context for the topic
            stream: Whether to stream the response token by token
            
        Returns/Yields:
            If stream=False: Dictionary with answer and complete flag
            If stream=True: Generator yielding tokens as they're generated
        """
        try:
            if context and not self._is_question_in_context(topic, context):
                if stream:
                    yield "Cannot generate bullet points for this topic as it is out of scope."
                    return
                return {'answer': "Cannot generate bullet points for this topic as it is out of scope.", 'complete': True}
            
            question_type = "bullet_points"
            max_word_count = self._determine_max_word_count(question_type)
            prompt = self._generate_prompt(topic, question_type, max_word_count)

            if not stream:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
                input_token_size = inputs["input_ids"].shape[1]

                max_new_tokens = round((max_word_count) * 1.3 + input_token_size)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                clean_response = self._get_reply(response)
                preprocessed_response = self._preprocess_bullet_points(clean_response)
                return {'answer': preprocessed_response, 'complete': self._is_response_complete(response)}
            
            else:
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
                input_token_size = inputs["input_ids"].shape[1]
                max_new_tokens = round((max_word_count) * 1.3 + input_token_size)

                # Create a streamer
                streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
                
                # Set up generation arguments
                generation_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "repetition_penalty": 1.2,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "streamer": streamer,
                }

                # Start generation in a separate thread
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                # Track the full response
                full_response = ""
                saw_think_close = False

                # Iterate through the streamed tokens
                for token in streamer:
                    # Add the token to the current response
                    full_response += token
                    
                    # Process think tags for the streaming output
                    if "</think>" in full_response and not saw_think_close:
                        # Once we see the closing tag, we only want to show content after it
                        saw_think_close = True
                        # Extract only what's after the last </think> tag
                        content_parts = full_response.split("</think>")
                        if len(content_parts) > 1:
                            clean_response = content_parts[-1].strip()
                            preprocessed = self._preprocess_bullet_points(clean_response)
                            yield preprocessed
                    elif saw_think_close:
                        # We're past the think tags, yield the full clean response each time
                        content_parts = full_response.split("</think>")
                        if len(content_parts) > 1:
                            clean_response = content_parts[-1].strip()
                            preprocessed = self._preprocess_bullet_points(clean_response)
                            yield preprocessed

                # If we never saw the closing think tag, clean the response
                if not saw_think_close and full_response:
                    clean_response = self._get_reply(full_response)
                    preprocessed = self._preprocess_bullet_points(clean_response)
                    yield preprocessed
        except Exception as e:
            error_msg = f"Error generating bullet points: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            if stream:
                yield error_msg
            else:
                return {'answer': error_msg, 'complete': False}

    def _preprocess_presentation(self, content):
        """
        Preprocess presentation content - handle slides separated by triple dashes.
        This preserves the original format while ensuring consistency for frontend rendering.
        """
        try:
            # Check if content is empty or too short
            if not content or len(content.strip()) < 10:
                return content
                
            # Case 1: Content uses triple dash slide separators
            if "---" in content:
                # Split by triple dash
                slides = re.split(r'\n\s*---\s*\n', content)
                processed_slides = []
                
                for slide in slides:
                    slide = slide.strip()
                    if slide:  # Only process non-empty slides
                        # Ensure each slide has a title (starts with #)
                        if not slide.startswith('#'):
                            # Find the first line and make it a title
                            first_line = slide.split('\n', 1)[0]
                            rest = slide[len(first_line):] if len(slide) > len(first_line) else ""
                            slide = f"# {first_line.strip()}{rest}"
                        
                        processed_slides.append(slide)
                
                # Rejoin with consistent separators
                if processed_slides:
                    return "\n---\n".join(processed_slides)
                return content
                
            # Case 2: Content uses <slide> tags
            elif "<slide>" in content:
                # Convert <slide> tags to triple dash format
                content = re.sub(r'<slide>\s*', '', content)
                content = re.sub(r'\s*</slide>', '', content)
                slides = content.split('\n\n')
                processed_slides = []
                
                for slide in slides:
                    slide = slide.strip()
                    if slide:  # Only process non-empty slides
                        processed_slides.append(slide)
                
                # Rejoin with consistent separators
                if processed_slides:
                    return "\n---\n".join(processed_slides)
                return content
                
            # Case 3: No explicit slide markers, try to identify slides by headings
            else:
                # Find all heading lines (# Title)
                headings = re.findall(r'(^|\n)#\s+(.+?)(\n|$)', content)
                
                if headings:
                    # Split content at each heading
                    slides = re.split(r'(^|\n)#\s+', content)
                    slides = slides[1:]  # Remove the first empty item
                    
                    processed_slides = []
                    for i, slide_content in enumerate(slides):
                        if slide_content.strip():
                            processed_slides.append(f"# {slide_content.strip()}")
                    
                    # Rejoin with consistent separators
                    if processed_slides:
                        return "\n---\n".join(processed_slides)
                
                # Case 4: No clear slide structure, try to create slides from paragraphs
                paragraphs = content.split('\n\n')
                if len(paragraphs) >= 3:  # At least 3 paragraphs to make a presentation
                    processed_slides = []
                    
                    # Make the first paragraph a title slide
                    processed_slides.append(f"# {paragraphs[0].strip()}")
                    
                    # Process remaining paragraphs as content slides
                    for i, para in enumerate(paragraphs[1:]):
                        if para.strip():
                            # If paragraph starts with a bullet point, assume it's a list slide
                            if para.strip().startswith('*'):
                                processed_slides.append(f"# Slide {i+1}\n\n{para.strip()}")
                            else:
                                title = f"Slide {i+1}"
                                processed_slides.append(f"# {title}\n\n{para.strip()}")
                    
                    # Rejoin with consistent separators
                    return "\n---\n".join(processed_slides)
                
                # Default: return original content if we can't identify slides
                return content
                
        except Exception as e:
            self.logger.warning(f"Error preprocessing presentation: {str(e)}")
            return content

if __name__ == "__main__":
    try:
        chatbot = FineTunedChatbot()

        def create_presentation():
            topic = "Displacement reaction in chemistry."

            start_time = time.time()
            presentation = chatbot.create_presentation(topic)
            end_time = time.time()
            print(presentation['answer'])

            print(f"\n\nTime taken: {end_time - start_time:.2f} seconds")

        def create_bullet_points():
            topic = "Key concepts in quantum mechanics"

            for token in chatbot.create_bullet_points(topic, stream=True):
                print('\033c', token, end='\r')

        def ask_question():
            question = "Explain how neural networks work"

            for token in chatbot.ask_stream(question):
                print('\033c', token, end='\r')

        # Choose which function to run
        create_presentation()
        # create_bullet_points()
        # ask_question()
        
    except Exception as e:
        print(f"Error in main: {str(e)}")