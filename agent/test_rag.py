#!/usr/bin/env python3
"""
Test script for the RAG implementation in the NVExperimentAgent.
This script simulates a conversation, saves embeddings, and then performs a RAG query.
"""

import os
import json
from agent import NVExperimentAgent

def main():
    print("=== Testing RAG Implementation ===")
    
    # Create a test agent
    agent = NVExperimentAgent()
    
    # Simulate a conversation
    print("\n1. Simulating a conversation...")
    
    # Add some test messages to the conversation history
    agent.conversation_history.append({"role": "user", "content": "Can you help me run an ESR experiment?"})
    agent.conversation_history.append({"role": "assistant", "content": "I'd be happy to help you run an ESR experiment. Let me guide you through the process."})
    agent.conversation_history.append({"role": "assistant", "content": "First, we need to read the default ESR configuration."})
    agent.conversation_history.append({"role": "assistant", "content": "(Read file) projects/configs/default_esr_config.json with content:\n{\"frequency_range\": [2.7e9, 3.0e9], \"power\": -10, \"points\": 100}"})
    agent.conversation_history.append({"role": "user", "content": "Can we modify the frequency range to 2.8-3.1 GHz?"})
    agent.conversation_history.append({"role": "assistant", "content": "Yes, we can modify the frequency range. I'll create a new configuration file for you."})
    
    # Simulate a plot analysis
    agent.conversation_history.append({"role": "assistant", "content": "VISION: projects/NVExperiment/runs/run_20250502_123456/data/ESR_plot.png"})
    agent.conversation_history.append({"role": "assistant", "content": "[System] Vision analysis result:\nThe ESR plot shows a dip at approximately 2.87 GHz, which indicates the presence of an NV center. The contrast is about 15%."})
    
    # Create a mock plot file to test plot detection
    os.makedirs(agent.data_dir, exist_ok=True)
    with open(os.path.join(agent.data_dir, "ESR_plot.png"), "w") as f:
        f.write("Mock plot file")
    
    # Save the conversation embeddings
    print("\n2. Saving conversation embeddings...")
    agent.save_conversation_embeddings()
    
    # Check if embeddings were created
    embeddings_files = [f for f in os.listdir(agent.embeddings_dir) if f.endswith('.json')]
    if embeddings_files:
        print(f"Successfully created embeddings file: {embeddings_files[0]}")
        
        # Load the embeddings file to verify its content
        with open(os.path.join(agent.embeddings_dir, embeddings_files[0]), 'r') as f:
            data = json.load(f)
            print(f"Embeddings file contains text and vector of dimension {len(data['embeddings'])}")
    else:
        print("Failed to create embeddings file")
        return
    
    # Test RAG query
    print("\n3. Testing RAG query...")
    query = "What was the ESR frequency range?"
    context = agent._get_rag_context(query)
    print(f"Query: '{query}'")
    print(f"Retrieved context:\n{context}")
    
    # Test plot detection
    print("\n4. Testing plot detection...")
    relevant_plots = agent._get_relevant_plots("Can you analyze the ESR results?")
    print(f"Relevant plots for 'Can you analyze the ESR results?': {[os.path.basename(p) for p in relevant_plots]}")
    
    # Test plot tracking
    print("\n5. Testing plot tracking...")
    analyzed_plots = agent._track_analyzed_plots()
    print(f"Analyzed plots: {analyzed_plots}")
    
    print("\n=== RAG Implementation Test Complete ===")

if __name__ == "__main__":
    main()
