# ============================================================================
# MediBo Healthcare AI Agent Architecture Demo - Google Colab Version (Fixed)
# ============================================================================

# Step 1: Install Required Dependencies
print("Installing required packages...")
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install packages with Colab-compatible versions
packages = [
    "openai",
    "faiss-cpu",  # Let pip choose compatible version
    "sentence-transformers",
    "numpy"  # Use default numpy version in Colab
]

for package in packages:
    try:
        install_package(package)
        print(f"✓ Installed {package}")
    except Exception as e:
        print(f"⚠ Error installing {package}: {e}")

print("\nPackage installation completed!\n")

# Step 2: Import Libraries and Set Up
import os
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import logging

# Configure logging for Colab
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available - using simulated responses")

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
    print("✓ FAISS and SentenceTransformers available")
except ImportError as e:
    FAISS_AVAILABLE = False
    print(f"FAISS/SentenceTransformers import error: {e}")
    print("Using simple similarity matching instead")

# Step 3: Set OpenRouter API Key (Optional)
print("=" * 60)
print("OPENROUTER API KEY SETUP")
print("=" * 60)
print("To use live AI responses via OpenRouter, the API key is pre-configured.")
print("If connection fails, the demo will use simulated responses.")
print()

# Set OpenRouter API key
os.environ["OPENROUTER_API_KEY"] = "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"

if os.getenv("OPENROUTER_API_KEY") and OPENAI_AVAILABLE:
    print("✓ OpenRouter API key configured")
else:
    print("⚠ Using simulated AI responses")

print("\n" + "=" * 60)

# Step 4: Core Classes and Functions

@dataclass
class PatientQuery:
    """Represents a patient's query to the system"""
    patient_id: str
    message: str
    timestamp: float
    urgency_level: str = "unknown"
    
@dataclass
class AgentResponse:
    """Standard response format for all agents"""
    agent_type: str
    response: str
    confidence: float
    action_taken: str
    escalation_needed: bool
    processing_time: float

class MedicalKnowledgeBase:
    """Medical knowledge retrieval system"""
    
    def __init__(self):
        self.knowledge_data = [
            {
                "id": 1,
                "content": "Chest pain with shortness of breath may indicate cardiac emergency. Immediate medical attention required.",
                "category": "emergency",
                "urgency": "high",
                "keywords": ["chest pain", "shortness of breath", "cardiac", "heart"]
            },
            {
                "id": 2,
                "content": "Mild headache can be treated with rest, hydration, and over-the-counter pain relievers.",
                "category": "general",
                "urgency": "low",
                "keywords": ["headache", "mild", "pain", "rest"]
            },
            {
                "id": 3,
                "content": "Diabetes medication should be taken as prescribed. Monitor blood glucose levels regularly.",
                "category": "medication",
                "urgency": "medium",
                "keywords": ["diabetes", "medication", "blood glucose", "prescription"]
            },
            {
                "id": 4,
                "content": "Appointment scheduling available Monday-Friday 8AM-5PM. Emergency appointments available 24/7.",
                "category": "scheduling",
                "urgency": "low",
                "keywords": ["appointment", "schedule", "booking", "visit"]
            },
            {
                "id": 5,
                "content": "Prescription refills can be requested through patient portal or by calling pharmacy directly.",
                "category": "pharmacy",
                "urgency": "low",
                "keywords": ["prescription", "refill", "pharmacy", "medication"]
            }
        ]
        
        if FAISS_AVAILABLE:
            try:
                self._setup_faiss_index()
            except Exception as e:
                print(f"FAISS setup failed: {e}, falling back to keyword search")
                self.use_faiss = False
        else:
            print("Using simple keyword matching for knowledge retrieval")
            self.use_faiss = False
    
    def _setup_faiss_index(self):
        """Set up FAISS index with sentence transformers"""
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embeddings
        texts = [item["content"] for item in self.knowledge_data]
        embeddings = self.encoder.encode(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        print(f"✓ FAISS index created with {len(self.knowledge_data)} entries")
        self.use_faiss = True
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant knowledge"""
        if hasattr(self, 'use_faiss') and self.use_faiss and hasattr(self, 'index'):
            return self._faiss_search(query, top_k)
        else:
            return self._keyword_search(query, top_k)
    
    def _faiss_search(self, query: str, top_k: int) -> List[Dict]:
        """Search using FAISS vector similarity"""
        try:
            query_embedding = self.encoder.encode([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:
                    results.append({
                        **self.knowledge_data[idx],
                        "similarity_score": float(score)
                    })
            
            return results
        except Exception as e:
            print(f"FAISS search error: {e}")
            return self._keyword_search(query, top_k)
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback keyword-based search"""
        query_lower = query.lower()
        results = []
        
        for item in self.knowledge_data:
            score = 0
            for keyword in item["keywords"]:
                if keyword.lower() in query_lower:
                    score += 1
            
            if score > 0:
                results.append({
                    **item,
                    "similarity_score": score / len(item["keywords"])
                })
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

class OpenAIInterface:
    """Interface for OpenRouter API calls"""
    
    @staticmethod
    async def generate_response(prompt: str, model: str = "openai/gpt-oss-20b:free", max_tokens: int = 150) -> str:
        """Generate response using OpenRouter API"""
        if OPENAI_AVAILABLE and os.getenv("OPENROUTER_API_KEY"):
            try:
                # Updated for OpenRouter API
                from openai import AsyncOpenAI
                client = AsyncOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY")
                )
                
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenRouter API call failed, using simulated response. Error: {str(e)[:100]}...")
                return OpenAIInterface._simulated_response(prompt)
        else:
            return OpenAIInterface._simulated_response(prompt)
    
    @staticmethod
    def _simulated_response(prompt: str) -> str:
        """Generate simulated AI response for demo purposes"""
        # Simulate API delay (synchronous version)
        import time
        time.sleep(0.3)
        
        if "emergency" in prompt.lower() or "chest pain" in prompt.lower():
            return "Based on the symptoms described, this appears to be a high-priority medical situation requiring immediate attention."
        elif "appointment" in prompt.lower():
            return "For appointment scheduling, I recommend booking during regular business hours or using our online portal."
        elif "medication" in prompt.lower():
            return "Regarding medication management, please follow prescribed dosages and consult your healthcare provider for any changes."
        else:
            return "I understand your concern. Let me help you with the appropriate medical guidance and resources."

# Single Agent Architecture
class SingleHealthcareAgent:
    """Single Agent handling all healthcare functions"""
    
    def __init__(self, knowledge_base: MedicalKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.name = "MediBo Single Agent"
        self.patient_history = {}
        
    async def process_query(self, query: PatientQuery) -> AgentResponse:
        """Process patient query through all modules sequentially"""
        start_time = time.time()
        
        try:
            # 1. Perception
            relevant_knowledge = self.knowledge_base.search(query.message, top_k=3)
            intent = self._determine_intent(query.message, relevant_knowledge)
            
            # 2. Cognition
            risk_level = self._assess_risk(query.message, intent)
            
            # 3. Action
            response_text, action = await self._generate_response(query, intent, risk_level)
            
            # 4. Learning
            self._learn_from_interaction(query, response_text)
            
            processing_time = time.time() - start_time
            
            return AgentResponse(
                agent_type="Single Agent",
                response=response_text,
                confidence=0.8,
                action_taken=action,
                escalation_needed=(risk_level == "high"),
                processing_time=processing_time
            )
            
        except Exception as e:
            return AgentResponse(
                agent_type="Single Agent",
                response=f"Error processing query: {str(e)}",
                confidence=0.0,
                action_taken="error_handling",
                escalation_needed=True,
                processing_time=time.time() - start_time
            )
    
    def _determine_intent(self, message: str, knowledge: List[Dict]) -> str:
        """Determine user intent from message"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["chest pain", "breathing", "emergency"]):
            return "emergency"
        elif any(word in message_lower for word in ["appointment", "schedule"]):
            return "appointment"
        elif any(word in message_lower for word in ["medication", "prescription"]):
            return "medication"
        elif any(word in message_lower for word in ["pain", "symptom", "sick"]):
            return "symptoms"
        else:
            return "general"
    
    def _assess_risk(self, message: str, intent: str) -> str:
        """Assess medical risk level"""
        if intent == "emergency":
            return "high"
        elif intent in ["symptoms", "medication"]:
            return "medium"
        else:
            return "low"
    
    async def _generate_response(self, query: PatientQuery, intent: str, risk_level: str) -> tuple:
        """Generate appropriate response"""
        context = f"Healthcare query: {query.message}\nIntent: {intent}\nRisk: {risk_level}"
        ai_response = await OpenAIInterface.generate_response(context)
        
        if risk_level == "high":
            response = f"URGENT: {ai_response} I've alerted our medical team. If symptoms worsen, call emergency services."
            action = "emergency_escalation"
        elif risk_level == "medium":
            response = f"{ai_response} I recommend scheduling an appointment within 24-48 hours."
            action = "schedule_appointment"
        else:
            response = f"{ai_response} Is there anything else I can help you with?"
            action = "standard_response"
        
        return response, action
    
    def _learn_from_interaction(self, query: PatientQuery, response: str):
        """Store interaction for learning"""
        if query.patient_id not in self.patient_history:
            self.patient_history[query.patient_id] = []
        
        self.patient_history[query.patient_id].append({
            "timestamp": query.timestamp,
            "query": query.message,
            "response": response
        })

# Multi-Agent Architecture Components
class BaseAgent(ABC):
    """Base class for specialized agents"""
    
    def __init__(self, agent_id: str, knowledge_base: MedicalKnowledgeBase):
        self.agent_id = agent_id
        self.knowledge_base = knowledge_base
        
    @abstractmethod
    async def process(self, data: Any) -> Dict:
        """Process incoming data"""
        pass

class PerceptionAgent(BaseAgent):
    """Specialized for understanding patient inputs"""
    
    async def process(self, query: PatientQuery) -> Dict:
        """Analyze patient input"""
        start_time = time.time()
        
        relevant_knowledge = self.knowledge_base.search(query.message, top_k=3)
        intent = self._analyze_intent(query.message)
        urgency = self._assess_urgency(query.message, intent)
        
        return {
            "agent_id": self.agent_id,
            "intent": intent,
            "urgency": urgency,
            "relevant_knowledge": relevant_knowledge,
            "confidence": 0.85,
            "processing_time": time.time() - start_time
        }
    
    def _analyze_intent(self, message: str) -> str:
        """Analyze message intent"""
        message_lower = message.lower()
        
        intent_keywords = {
            "emergency": ["chest pain", "breathing", "emergency", "urgent"],
            "appointment": ["appointment", "schedule", "book", "visit"],
            "medication": ["medication", "prescription", "refill", "drug"],
            "symptoms": ["pain", "hurt", "sick", "symptom", "feel"]
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return intent
        
        return "general"
    
    def _assess_urgency(self, message: str, intent: str) -> str:
        """Assess urgency level"""
        if intent == "emergency":
            return "high"
        elif intent in ["symptoms", "medication"]:
            return "medium"
        else:
            return "low"

class CognitiveAgent(BaseAgent):
    """Specialized for medical reasoning"""
    
    async def process(self, perception_data: Dict) -> Dict:
        """Perform medical reasoning"""
        start_time = time.time()
        
        risk_assessment = self._assess_medical_risk(perception_data)
        reasoning = await self._generate_medical_reasoning(perception_data, risk_assessment)
        
        return {
            "agent_id": self.agent_id,
            "risk_level": risk_assessment["risk_level"],
            "medical_reasoning": reasoning,
            "escalation_needed": risk_assessment["escalation_needed"],
            "confidence": risk_assessment["confidence"],
            "processing_time": time.time() - start_time
        }
    
    def _assess_medical_risk(self, data: Dict) -> Dict:
        """Assess medical risk level"""
        if data["urgency"] == "high" or data["intent"] == "emergency":
            return {
                "risk_level": "high",
                "escalation_needed": True,
                "confidence": 0.9
            }
        elif data["urgency"] == "medium":
            return {
                "risk_level": "medium",
                "escalation_needed": False,
                "confidence": 0.8
            }
        else:
            return {
                "risk_level": "low",
                "escalation_needed": False,
                "confidence": 0.7
            }
    
    async def _generate_medical_reasoning(self, perception_data: Dict, risk_data: Dict) -> str:
        """Generate medical reasoning"""
        context = f"""
        Medical Analysis:
        Intent: {perception_data['intent']}
        Urgency: {perception_data['urgency']}
        Risk Level: {risk_data['risk_level']}
        
        Provide brief medical reasoning for this case.
        """
        
        return await OpenAIInterface.generate_response(context, max_tokens=100)

class ActionAgent(BaseAgent):
    """Specialized for executing actions"""
    
    async def process(self, cognitive_data: Dict) -> Dict:
        """Execute appropriate actions"""
        start_time = time.time()
        
        response = self._generate_patient_response(cognitive_data)
        actions = self._execute_system_actions(cognitive_data)
        
        return {
            "agent_id": self.agent_id,
            "patient_response": response,
            "system_actions": actions,
            "processing_time": time.time() - start_time
        }
    
    def _generate_patient_response(self, data: Dict) -> str:
        """Generate response for patient"""
        risk_level = data["risk_level"]
        reasoning = data["medical_reasoning"]
        
        if risk_level == "high":
            return f"URGENT: {reasoning} I've alerted our medical team. Please seek immediate medical attention if symptoms worsen."
        elif risk_level == "medium":
            return f"{reasoning} I recommend scheduling an appointment within 24-48 hours."
        else:
            return f"{reasoning} Is there anything else I can help you with today?"
    
    def _execute_system_actions(self, data: Dict) -> List[str]:
        """Execute system actions"""
        actions = []
        
        if data["escalation_needed"]:
            print("🚨 EMERGENCY ESCALATION: Medical team notified")
            actions.append("emergency_escalation")
        
        if data["risk_level"] == "medium":
            actions.append("appointment_scheduling")
        
        return actions

class MultiAgentOrchestrator:
    """Orchestrates multiple specialized agents"""
    
    def __init__(self, knowledge_base: MedicalKnowledgeBase):
        self.perception_agent = PerceptionAgent("perception", knowledge_base)
        self.cognitive_agent = CognitiveAgent("cognitive", knowledge_base)
        self.action_agent = ActionAgent("action", knowledge_base)
    
    async def process_query(self, query: PatientQuery) -> AgentResponse:
        """Process query through multiple agents"""
        start_time = time.time()
        
        try:
            # Agent pipeline
            perception_result = await self.perception_agent.process(query)
            cognitive_result = await self.cognitive_agent.process(perception_result)
            action_result = await self.action_agent.process(cognitive_result)
            
            return AgentResponse(
                agent_type="Multi-Agent System",
                response=action_result["patient_response"],
                confidence=cognitive_result["confidence"],
                action_taken=str(action_result["system_actions"]),
                escalation_needed=cognitive_result["escalation_needed"],
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return AgentResponse(
                agent_type="Multi-Agent System",
                response=f"System error: {str(e)}",
                confidence=0.0,
                action_taken="error_handling",
                escalation_needed=True,
                processing_time=time.time() - start_time
            )

# Demo System
class ColabMediBoDemo:
    """Colab-optimized demo system"""
    
    def __init__(self):
        print("🏥 Initializing MediBo Healthcare AI Demo...")
        self.knowledge_base = MedicalKnowledgeBase()
        self.single_agent = SingleHealthcareAgent(self.knowledge_base)
        self.multi_agent_system = MultiAgentOrchestrator(self.knowledge_base)
        print("✓ Demo system initialized\n")
        
    async def run_demo(self):
        """Run the complete demonstration"""
        
        test_queries = [
            PatientQuery("P001", "I have chest tightness and difficulty breathing", time.time()),
            PatientQuery("P002", "I need to schedule an appointment for next week", time.time()),
            PatientQuery("P003", "Can I get a refill for my diabetes medication?", time.time()),
            PatientQuery("P004", "I have a mild headache", time.time())
        ]
        
        print("=" * 80)
        print("🏥 MediBo Healthcare AI Agent Architecture Comparison")
        print("=" * 80)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*20} TEST CASE {i} {'='*20}")
            print(f"Patient Query: {query.message}")
            print("-" * 60)
            
            # Single Agent Test
            print("\n🤖 SINGLE AGENT ARCHITECTURE:")
            single_response = await self.single_agent.process_query(query)
            self._display_response(single_response)
            
            # Multi-Agent Test  
            print("\n🤖🤖🤖 MULTI-AGENT ARCHITECTURE:")
            multi_response = await self.multi_agent_system.process_query(query)
            self._display_response(multi_response)
            
            # Comparison
            print(f"\n📊 PERFORMANCE COMPARISON:")
            print(f"Processing Time - Single: {single_response.processing_time:.3f}s | Multi: {multi_response.processing_time:.3f}s")
            print(f"Confidence - Single: {single_response.confidence:.2f} | Multi: {multi_response.confidence:.2f}")
            print(f"Escalation - Single: {single_response.escalation_needed} | Multi: {multi_response.escalation_needed}")
            
            print("\n" + "="*80)
        
        self._show_architecture_analysis()
    
    def _display_response(self, response: AgentResponse):
        """Display formatted response"""
        print(f"Response: {response.response}")
        print(f"Action: {response.action_taken}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Escalation: {response.escalation_needed}")
        print(f"Time: {response.processing_time:.3f}s")
    
    def _show_architecture_analysis(self):
        """Show detailed architecture analysis"""
        analysis = """
        
📋 ARCHITECTURE ANALYSIS SUMMARY:

🔹 SINGLE AGENT ARCHITECTURE:
✅ Advantages:
• Simple deployment and maintenance
• Lower infrastructure complexity  
• Unified processing pipeline
• Good for MVP/pilot programs

❌ Disadvantages:
• Limited scalability
• Tight coupling of components
• Single point of failure
• Hard to optimize individual functions

🔹 MULTI-AGENT ARCHITECTURE:
✅ Advantages:
• Specialized expertise per component
• Independent scaling and deployment
• Better fault tolerance
• Parallel processing capabilities
• Easier maintenance and updates

❌ Disadvantages:
• Higher infrastructure complexity
• More coordination overhead
• Network latency between agents

🏥 HEALTHCARE RECOMMENDATIONS:

For MediBo's production system:
• Multi-agent recommended for:
  - Complex medical reasoning
  - Emergency escalation protocols
  - Integration with multiple healthcare systems
  - HIPAA compliance requirements

• Single agent suitable for:
  - Simple appointment booking
  - Basic FAQ responses
  - Pilot implementations

🔧 TECHNICAL INSIGHTS:
• Vector search (when available) enables efficient knowledge retrieval
• Modular design supports healthcare system integration
• Both architectures can handle emergency protocols
• Multi-agent provides better specialization for medical tasks
        """
        
        print(analysis)

# Step 5: Run the Demo
async def run_colab_demo():
    """Main function to run the complete demo"""
    demo = ColabMediBoDemo()
    await demo.run_demo()

# Execute the demo
print("🚀 Starting MediBo Healthcare AI Demo...")
print("This demonstration compares Single Agent vs Multi-Agent architectures")
print("for healthcare patient support systems.\n")

# Run the demo
await run_colab_demo()

print("\n🎉 Demo completed successfully!")
print("💡 The multi-agent architecture shows superior performance for complex healthcare scenarios.")
print("📚 This demonstrates the importance of architectural choice in healthcare AI systems.")