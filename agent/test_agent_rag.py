#!/usr/bin/env python3
"""
Interactive test script for the NVExperimentAgent with RAG functionality.
This script allows you to interact with the agent and see the RAG functionality in action.
"""

import os
import sys
from agent import NVExperimentAgent

def main():
    print("=== NV Experiment Agent with RAG Testing ===")
    
    # Create a test agent
    agent = NVExperimentAgent()
    
    # Create test data directory and mock plot files
    os.makedirs(agent.data_dir, exist_ok=True)
    
    # Create mock plot files
    plot_files = ["ESR_plot.png", "GalvoScan_plot.png", "FindNV_plot.png", "Optimization_plot.png"]
    for plot_file in plot_files:
        with open(os.path.join(agent.data_dir, plot_file), "w") as f:
            f.write(f"Mock {plot_file} file")
    
    print(f"\nCreated mock plot files in {agent.data_dir}")
    
    # Create a sample embedding to test RAG functionality
    sample_conversation = [
        {"role": "user", "content": "Can you help me run an ESR experiment?"},
        {"role": "assistant", "content": "I'd be happy to help you run an ESR experiment. Let me guide you through the process."},
        {"role": "assistant", "content": "First, we need to read the default ESR configuration."},
        {"role": "assistant", "content": "(Read file) projects/configs/default_esr_config.json with content:\n{\"frequency_range\": [2.7e9, 3.0e9], \"power\": -10, \"points\": 100}"},
        {"role": "user", "content": "Can we modify the frequency range to 2.8-3.1 GHz?"},
        {"role": "assistant", "content": "Yes, we can modify the frequency range. I'll create a new configuration file for you."},
        {"role": "assistant", "content": "VISION: projects/NVExperiment/runs/run_20250502_123456/data/ESR_plot.png"},
        {"role": "assistant", "content": "[System] Vision analysis result:\nThe ESR plot shows a dip at approximately 2.87 GHz, which indicates the presence of an NV center. The contrast is about 15%."}
    ]
    
    # Add the sample conversation to the agent's history
    agent.conversation_history = sample_conversation
    
    # Save the sample conversation embeddings
    print("\nSaving sample conversation embeddings...")
    agent.save_conversation_embeddings()
    
    print("\n=== Interactive Agent Testing ===")
    print("Type your messages to interact with the agent.")
    print("The agent will perform RAG queries for each of your inputs.")
    print("Type 'exit' or 'quit' to end the test.")
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Saving conversation embeddings before exit...")
                agent.save_conversation_embeddings()
                print("Test completed. Goodbye!")
                break
            
            # Process the user input through the agent
            agent.handle_user_input(user_input)
            
        except KeyboardInterrupt:
            print("\nTest interrupted. Saving conversation embeddings...")
            agent.save_conversation_embeddings()
            print("Goodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            continue

if __name__ == "__main__":
    main()
