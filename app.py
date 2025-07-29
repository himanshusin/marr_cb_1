#%% Section 1: Import Dependencies
import streamlit as st
import boto3
import json
import os
import traceback
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker import get_execution_role
import time
from typing import Dict, List


def log_exception(context: str, exc: Exception) -> None:
    """Display verbose error information in the app."""
    st.error(f"{context}: {exc}")
    st.text(traceback.format_exc())

#%% Section 2: Configuration and Setup
# Page config
st.set_page_config(
    page_title="Marriott Credit Card Assistant",
    page_icon="🏨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Model configuration
MODEL_ID = "meta-textgeneration-llama-2-7b-f"
MODEL_VERSION = "*"
ACCEPT_EULA = True  # Set to True to accept End User License Agreement

# Marriott Credit Card Sales Bot Configuration
SYSTEM_PROMPT = """You are a professional and knowledgeable Marriott Credit Card sales assistant. Your role is to help customers understand the benefits of Marriott Bonvoy credit cards and guide them through the application process.

PERSONALITY:
- Professional, friendly, and enthusiastic about travel
- Knowledgeable about Marriott properties and rewards
- Helpful and patient with customer questions
- Focus on value and benefits that match customer needs

KEY PRODUCTS TO PROMOTE:
1. Marriott Bonvoy Boundless Credit Card
   - 3X points at Marriott properties
   - 2X points on all other purchases
   - Annual free night award (up to 35,000 points)
   - No foreign transaction fees
   
2. Marriott Bonvoy Bold Credit Card
   - 2X points at Marriott properties
   - 1X point on all other purchases
   - No annual fee
   - Good starter card for new travelers

3. Marriott Bonvoy Brilliant American Express Card
   - 6X points at Marriott properties
   - 3X points on dining and travel
   - Annual free night award (up to 50,000 points)
   - Premium benefits like Priority Pass lounge access

CONVERSATION GUIDELINES:
1. Always greet customers warmly and ask how you can help with their travel and credit card needs
2. Ask about their travel habits to recommend the best card
3. Highlight relevant benefits based on their interests
4. Address concerns professionally and honestly
5. Provide clear next steps for application
6. Stay focused on Marriott credit cards - politely redirect off-topic conversations

GUARDRAILS:
- Never guarantee approval - explain that applications are subject to credit approval
- Don't provide specific interest rates or fees without noting they may vary
- Always mention terms and conditions apply
- Don't give financial advice beyond credit card benefits
- Respectfully decline to discuss competitor credit cards in detail
- Don't make promises about point values or redemption rates that may change
- Always remind customers to review full terms before applying

PROHIBITED TOPICS:
- Other hotel chains' credit cards (redirect to Marriott options)
- Personal financial advice beyond credit card selection
- Specific credit scores or approval odds
- Political topics or controversial subjects
- Technical discussions unrelated to credit cards or travel

Remember: Your goal is to help customers find the right Marriott credit card for their travel lifestyle while being honest, helpful, and professional."""

WELCOME_MESSAGE = """🏨 Welcome to Marriott Credit Card Services! 

I'm here to help you discover the perfect Marriott Bonvoy credit card for your travel lifestyle. Whether you're a frequent business traveler, vacation enthusiast, or just starting your travel journey, we have a card that can enhance your experiences.

How can I assist you today? I can help you:
✈️ Learn about our credit card benefits
🏨 Find the right card for your travel habits  
💳 Understand rewards and perks
🎯 Start your application process

What interests you most about travel rewards?"""

#%% Section 3: Initialize Session State
def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]
    if "model_deployed" not in st.session_state:
        st.session_state.model_deployed = False
    if "predictor" not in st.session_state:
        st.session_state.predictor = None
    if "deployment_status" not in st.session_state:
        st.session_state.deployment_status = "Not connected"
    if "endpoint_name" not in st.session_state:
        st.session_state.endpoint_name = ""
    if "connection_method" not in st.session_state:
        st.session_state.connection_method = "existing"
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = []

#%% Section 4: Model Deployment and Connection Functions
@st.cache_resource
def setup_aws_session():
    """Setup AWS session and SageMaker"""
    try:
        # Initialize boto3 session with default region
        session = boto3.Session(region_name="us-east-1")
        
        # Get AWS credentials info for display
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        
        return session, identity
    except Exception as e:
        log_exception("AWS Setup Error", e)
        return None, None

def get_existing_endpoints():
    """Get list of existing SageMaker endpoints"""
    try:
        session = boto3.Session(region_name="us-east-1")
        sagemaker_client = session.client('sagemaker')
        
        response = sagemaker_client.list_endpoints(
            StatusEquals='InService',
            MaxResults=50
        )
        
        endpoints = [ep['EndpointName'] for ep in response['Endpoints']]
        return endpoints
    except Exception as e:
        log_exception("Error fetching endpoints", e)
        return []

def connect_to_existing_endpoint(endpoint_name: str):
    """Connect to an existing SageMaker endpoint"""
    try:
        from sagemaker.predictor import Predictor
        from sagemaker.serializers import JSONSerializer
        from sagemaker.deserializers import JSONDeserializer
        
        # Create predictor for existing endpoint
        predictor = Predictor(
            endpoint_name=endpoint_name,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
        
        # Test the connection with a simple request
        test_payload = {
            "inputs": "Hello",
            "parameters": {
                "max_new_tokens": 10,
                "temperature": 0.7
            }
        }
        
        with st.spinner("Testing connection to endpoint..."):
            test_response = predictor.predict(test_payload)
        
        return predictor
    except Exception as e:
        log_exception("Error connecting to endpoint", e)
        return None

def deploy_new_model():
    """Deploy a new JumpStart model"""
    try:
        with st.spinner("Deploying new model... This may take 5-10 minutes."):
            # Create model instance
            model = JumpStartModel(
                model_id=MODEL_ID,
                model_version=MODEL_VERSION
            )
            
            # Deploy model
            predictor = model.deploy(
                initial_instance_count=1,
                instance_type="ml.g5.xlarge",  # GPU instance for better performance
                accept_eula=ACCEPT_EULA
            )
            
            return predictor
    except Exception as e:
        log_exception("Model Deployment Error", e)
        return None

def generate_response(predictor, user_message: str, max_length: int = 200) -> str:
    """Generate response from the deployed model with Marriott credit card sales context"""
    try:
        # Build conversation context
        conversation_history = ""
        if st.session_state.conversation_context:
            for msg in st.session_state.conversation_context[-6:]:  # Keep last 6 messages for context
                conversation_history += f"Customer: {msg['user']}\nAssistant: {msg['assistant']}\n\n"
        
        # Create the full prompt with system instructions and conversation context
        full_prompt = f"""{SYSTEM_PROMPT}

Previous conversation:
{conversation_history}

Customer: {user_message}
Assistant: """
        
        # Prepare the payload
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": max_length,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "stop": ["Customer:", "Human:", "\n\nCustomer:", "\n\nHuman:"]
            }
        }
        
        # Get prediction
        response = predictor.predict(payload)

        # Extract generated text from various possible formats
        generated_text = ""
        try:
            if isinstance(response, list) and len(response) > 0:
                first_item = response[0]
                if isinstance(first_item, dict):
                    generated_text = first_item.get("generated_text", "")
                else:
                    generated_text = str(first_item)
            elif isinstance(response, dict):
                generated_text = response.get("generated_text", "")
            elif isinstance(response, str):
                # Some endpoints return a JSON string
                try:
                    data = json.loads(response)
                    if isinstance(data, list) and len(data) > 0:
                        generated_text = data[0].get("generated_text", "")
                    elif isinstance(data, dict):
                        generated_text = data.get("generated_text", "")
                    else:
                        generated_text = response
                except json.JSONDecodeError:
                    generated_text = response
            else:
                generated_text = str(response)
        except Exception:
            generated_text = str(response)
        
        # Clean up the response
        if generated_text.startswith(full_prompt):
            generated_text = generated_text[len(full_prompt):].strip()
        
        # Apply guardrails and content filtering
        filtered_response = apply_guardrails(generated_text)
        
        # Store conversation context
        st.session_state.conversation_context.append({
            "user": user_message,
            "assistant": filtered_response
        })
        
        # Keep only last 10 conversation pairs to manage memory
        if len(st.session_state.conversation_context) > 10:
            st.session_state.conversation_context = st.session_state.conversation_context[-10:]
        
        return filtered_response if filtered_response else "I'm here to help you with Marriott credit cards. What would you like to know about our travel rewards program?"
        
    except Exception as e:
        log_exception("Prediction Error", e)
        return (
            "I apologize, but I'm having trouble processing your request right now. "
            "Let me help you with information about our Marriott credit cards. "
            "What specific benefits are you interested in learning about?"
        )

def apply_guardrails(response: str) -> str:
    """Apply content guardrails and ensure response stays on topic"""
    
    # Check for prohibited content or off-topic responses
    prohibited_keywords = [
        "chase sapphire", "amex platinum", "capital one", "citi", "discover",
        "stock market", "investment", "cryptocurrency", "bitcoin", "politics",
        "medical advice", "legal advice", "guarantee approval", "guaranteed points"
    ]
    
    response_lower = response.lower()
    
    # If response contains prohibited content, redirect
    for keyword in prohibited_keywords:
        if keyword in response_lower:
            return "I focus specifically on Marriott credit cards and travel rewards. Let me help you find the perfect Marriott Bonvoy card for your travel needs. Would you like to hear about our current card options and their benefits?"
    
    # Ensure response is related to Marriott/credit cards/travel
    marriott_keywords = [
        "marriott", "bonvoy", "credit card", "points", "travel", "hotel", "rewards",
        "benefits", "application", "annual fee", "boundless", "bold", "brilliant"
    ]
    
    has_relevant_content = any(keyword in response_lower for keyword in marriott_keywords)
    
    if not has_relevant_content and len(response) > 50:
        return "I'm here to help you with Marriott Bonvoy credit cards and travel rewards. What specific aspect of our credit cards would you like to learn more about? I can explain benefits, rewards, or help you choose the right card."
    
    # Clean up any inappropriate guarantees
    response = response.replace("guaranteed", "potential")
    response = response.replace("promise", "typically offer")
    response = response.replace("will definitely", "may")
    
    # Ensure compliance disclaimer for financial products
    if "apply" in response_lower or "application" in response_lower:
        if "terms and conditions" not in response_lower:
            response += " Please note that all applications are subject to credit approval and terms and conditions apply."
    
    return response

#%% Section 5: UI Components
def render_sidebar():
    """Render the sidebar with controls and info"""
    st.sidebar.title("🏨 Marriott Credit Card Assistant")
    
    # AWS Info
    session, identity = setup_aws_session()
    if identity:
        st.sidebar.success("✅ AWS Connected")
        st.sidebar.info(f"Region: {session.region_name}")
    else:
        st.sidebar.error("❌ AWS Connection Failed")
        return False
    
    # Model Connection Method
    st.sidebar.subheader("🔗 Connection Method")
    connection_method = st.sidebar.radio(
        "Choose how to connect:",
        ["existing", "deploy_new"],
        format_func=lambda x: "Use Existing Endpoint" if x == "existing" else "Deploy New Model",
        key="connection_method"
    )
    
    # Model Info
    st.sidebar.subheader("📊 Model Information")
    st.sidebar.info(f"Model: {MODEL_ID}")
    st.sidebar.info(f"Status: {st.session_state.deployment_status}")
    
    # Connection/Deployment controls
    st.sidebar.subheader("🚀 Model Connection")
    
    if not st.session_state.model_deployed:
        if connection_method == "existing":
            # Option 1: Connect to existing endpoint
            st.sidebar.markdown("**Connect to Existing Endpoint**")
            
            # Get list of available endpoints
            available_endpoints = get_existing_endpoints()
            
            if available_endpoints:
                selected_endpoint = st.sidebar.selectbox(
                    "Select an endpoint:",
                    [""] + available_endpoints,
                    help="Choose from your existing SageMaker endpoints"
                )
                
                if selected_endpoint and st.sidebar.button("🔗 Connect to Endpoint", type="primary"):
                    predictor = connect_to_existing_endpoint(selected_endpoint)
                    if predictor:
                        st.session_state.predictor = predictor
                        st.session_state.model_deployed = True
                        st.session_state.deployment_status = f"Connected to: {selected_endpoint}"
                        st.session_state.endpoint_name = selected_endpoint
                        st.sidebar.success("Connected to existing endpoint!")
                        st.rerun()
                    else:
                        st.sidebar.error("Failed to connect to the selected endpoint. Check logs for details.")
            else:
                st.sidebar.warning("No active endpoints found")
                st.sidebar.info("💡 Create an endpoint first or switch to 'Deploy New Model'")
            
            # Manual endpoint name input
            st.sidebar.markdown("**Or enter endpoint name manually:**")
            manual_endpoint = st.sidebar.text_input(
                "Endpoint name:",
                placeholder="jumpstart-dft-meta-textgeneration-llama-2-7b-f",
                help="Enter the exact name of your SageMaker endpoint"
            )
            
            if manual_endpoint and st.sidebar.button("🔗 Connect Manually"):
                predictor = connect_to_existing_endpoint(manual_endpoint)
                if predictor:
                    st.session_state.predictor = predictor
                    st.session_state.model_deployed = True
                    st.session_state.deployment_status = f"Connected to: {manual_endpoint}"
                    st.session_state.endpoint_name = manual_endpoint
                    st.sidebar.success("Connected to endpoint!")
                    st.rerun()
        
        else:
            # Option 2: Deploy new model
            st.sidebar.markdown("**Deploy New Model**")
            st.sidebar.warning("⚠️ This will take 5-10 minutes and incur costs")
            
            if st.sidebar.button("🚀 Deploy New Model", type="primary"):
                predictor = deploy_new_model()
                if predictor:
                    st.session_state.predictor = predictor
                    st.session_state.model_deployed = True
                    st.session_state.deployment_status = f"Deployed: {predictor.endpoint_name}"
                    st.session_state.endpoint_name = predictor.endpoint_name
                    st.sidebar.success("Model deployed successfully!")
                    st.rerun()
    
    else:
        # Model is connected/deployed
        st.sidebar.success("✅ Assistant is ready!")
        st.sidebar.info(f"Endpoint: {st.session_state.endpoint_name}")
        
        if st.sidebar.button("🔌 Disconnect"):
            # Only delete endpoint if we deployed it (not if we connected to existing)
            if st.session_state.connection_method == "deploy_new" and st.session_state.predictor:
                try:
                    with st.spinner("Deleting endpoint..."):
                        st.session_state.predictor.delete_endpoint()
                    st.sidebar.success("Endpoint deleted!")
                except Exception as e:
                    log_exception("Error deleting endpoint", e)
            
            # Reset connection state
            st.session_state.model_deployed = False
            st.session_state.predictor = None
            st.session_state.deployment_status = "Not connected"
            st.session_state.endpoint_name = ""
            st.rerun()
    
    # Quick Info Panel
    st.sidebar.subheader("💳 Quick Card Info")
    with st.sidebar.expander("Marriott Credit Cards"):
        st.markdown("""
        **Boundless Card:**
        • 3X points at Marriott
        • Free night award annually
        • No foreign transaction fees
        
        **Bold Card:**
        • 2X points at Marriott  
        • No annual fee
        • Great starter card
        
        **Brilliant Card:**
        • 6X points at Marriott
        • Premium travel benefits
        • Priority Pass access
        """)
    
    # Chat controls
    st.sidebar.subheader("💬 Chat Controls")
    if st.sidebar.button("🔄 Reset Conversation"):
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
        st.session_state.conversation_context = []
        st.rerun()
    
    # Settings
    st.sidebar.subheader("⚙️ Response Settings")
    max_length = st.sidebar.slider("Max Response Length", 100, 400, 250)
    
    return True, max_length

def render_chat_interface():
    """Render the main chat interface"""
    st.title("🏨 Marriott Credit Card Assistant")
    st.markdown("*Your personal guide to Marriott Bonvoy credit cards and travel rewards*")
    st.markdown("---")
    
    # Check if model is connected/deployed
    if not st.session_state.model_deployed:
        st.warning("⚠️ Please connect to the assistant first using the sidebar controls.")
        
        # Show helpful instructions based on connection method
        if st.session_state.connection_method == "existing":
            st.info("💡 Select an existing endpoint from the dropdown or enter the endpoint name manually in the sidebar.")
            st.markdown("""
            **Quick Start with Existing Endpoint:**
            1. Choose 'Use Existing Endpoint' in the sidebar
            2. Select your endpoint from the dropdown or enter it manually
            3. Click 'Connect to Endpoint'
            4. Start helping customers with Marriott credit cards!
            """)
        else:
            st.info("💡 Click the '🚀 Deploy New Model' button in the sidebar to get started.")
            st.markdown("""
            **Deploy New Model:**
            1. Choose 'Deploy New Model' in the sidebar
            2. Click 'Deploy New Model' (takes 5-10 minutes)
            3. Wait for deployment to complete
            4. Start helping customers with Marriott credit cards!
            """)
        
        # Show sample conversation starters
        st.subheader("💬 Sample Customer Questions:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - "What Marriott credit cards do you offer?"
            - "I travel for business monthly, which card is best?"
            - "What are the annual fees and benefits?"
            - "How do I earn and redeem Bonvoy points?"
            """)
        with col2:
            st.markdown("""
            - "Can you help me apply for a card?"
            - "What's the difference between your cards?"
            - "Do you have cards with no annual fee?"
            - "What hotels accept Bonvoy points?"
            """)
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about Marriott credit cards and travel rewards..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(
                    st.session_state.predictor, 
                    prompt, 
                    max_length=st.session_state.get('max_length', 200)
                )
            st.markdown(response)
        
        # Add assistant response to messages
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Quick action buttons for common questions
    st.markdown("---")
    st.subheader("🚀 Quick Questions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💳 Compare Cards", help="See all Marriott credit card options"):
            quick_question = "Can you compare all the Marriott credit cards and their benefits?"
            st.session_state.messages.append({"role": "user", "content": quick_question})
            with st.chat_message("user"):
                st.markdown(quick_question)
            with st.chat_message("assistant"):
                with st.spinner("Comparing card options..."):
                    response = generate_response(st.session_state.predictor, quick_question, st.session_state.get('max_length', 250))
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    with col2:
        if st.button("✈️ Travel Benefits", help="Learn about travel perks and rewards"):
            quick_question = "What travel benefits and perks do Marriott credit cards offer?"
            st.session_state.messages.append({"role": "user", "content": quick_question})
            with st.chat_message("user"):
                st.markdown(quick_question)
            with st.chat_message("assistant"):
                with st.spinner("Finding travel benefits..."):
                    response = generate_response(st.session_state.predictor, quick_question, st.session_state.get('max_length', 250))
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    with col3:
        if st.button("📝 Apply Now", help="Start your application process"):
            quick_question = "How can I apply for a Marriott credit card and what do I need?"
            st.session_state.messages.append({"role": "user", "content": quick_question})
            with st.chat_message("user"):
                st.markdown(quick_question)
            with st.chat_message("assistant"):
                with st.spinner("Preparing application information..."):
                    response = generate_response(st.session_state.predictor, quick_question, st.session_state.get('max_length', 250))
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

#%% Section 6: Main Application
def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar and get settings
    sidebar_result = render_sidebar()
    if isinstance(sidebar_result, tuple):
        aws_connected, max_length = sidebar_result
        st.session_state.max_length = max_length
    else:
        aws_connected = sidebar_result
    
    if not aws_connected:
        st.error("❌ AWS connection failed. Please check your credentials.")
        st.info("Make sure you have:")
        st.markdown("""
        - AWS credentials configured (via AWS CLI, environment variables, or IAM role)
        - Appropriate SageMaker permissions
        - Access to the specified model in SageMaker JumpStart
        """)
        return
    
    # Render main chat interface
    render_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip**: This chatbot uses AWS SageMaker JumpStart with Meta's Llama 2 model. "
        "Make sure you have the necessary AWS permissions and accept the model's EULA."
    )

#%% Section 7: Entry Point
if __name__ == "__main__":
    main()
