# Marriott Credit Card Assistant

This repository contains a Streamlit application that serves as a chatbot for Marriott credit card inquiries. The assistant uses Amazon SageMaker JumpStart with Meta's Llama 2 model to provide responses.

## Running the App

1. Install the required dependencies:
   ```bash
   pip install streamlit boto3 sagemaker -U
   ```
   Ensure you have AWS credentials configured with access to SageMaker.

2. Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```
   The app will open in your default browser.

## UI Samples

Below are some samples of the interface you will see when the app is running.

### Sidebar
```
🏨 Marriott Credit Card Assistant

⚙️ Settings
  ▸ Connect to existing endpoint or deploy a new model
  ▸ Adjust maximum response length

💳 Quick Card Info
  • Boundless Card – 3X points at Marriott
  • Bold Card – no annual fee
  • Brilliant Card – premium travel benefits
```

### Main Chat Window
```
🏨 Marriott Credit Card Assistant
*Your personal guide to Marriott Bonvoy credit cards and travel rewards*

User: "Which card is best for frequent travelers?"
Assistant: "For frequent travelers, the Marriott Bonvoy Brilliant card offers 6X points at Marriott properties and premium perks like Priority Pass lounge access."
```

### Quick Actions
```
[💳 Compare Cards]   [✈️ Travel Benefits]   [📝 Apply Now]
```

These buttons provide shortcuts to common requests such as comparing all cards or opening the Marriott credit card application page.

---

Feel free to customize the UI further to match your branding or add new features.
