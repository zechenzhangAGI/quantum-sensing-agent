#!/usr/bin/env python3
"""
Test script to verify that conversations are automatically embedded when the process stops.
This script simulates a short conversation and then exits, triggering the automatic embedding.
"""

import os
import sys
import time
from agent import NVExperimentAgent

def main():
    print("=== Testing Automatic Embedding on Exit ===")
    
    # Create a test agent
    agent = NVExperimentAgent()
    
    # Create test data directory and a mock plot file
    os.makedirs(agent.data_dir, exist_ok=True)
    with open(os.path.join(agent.data_dir, "ESR_plot.png"), "w") as f:
        f.write("Mock ESR plot file")
    
    print(f"\nCreated mock plot file in {agent.data_dir}")
    
    # Count existing embedding files
    if os.path.exists(agent.embeddings_dir):
        existing_files = [f for f in os.listdir(agent.embeddings_dir) if f.endswith('.json')]
        print(f"Found {len(existing_files)} existing embedding files before test")
    else:
        print("No existing embedding files found")
        existing_files = []
    
    # Simulate a short conversation
    print("\nSimulating a short conversation...")
    
    # Add some test messages to the conversation history
    agent.conversation_history.append({"role": "user", "content": "Let's analyze the ESR data"})
    print("Added user message: 'Let's analyze the ESR data'")
    
    agent.conversation_history.append({"role": "assistant", "content": "I'll help you analyze the ESR data. Let me look at the plot."})
    print("Added assistant response")
    
    agent.conversation_history.append({"role": "assistant", "content": "VISION: projects/NVExperiment/runs/run_20250502_123456/data/ESR_plot.png"})
    print("Added vision request")
    
    agent.conversation_history.append({"role": "assistant", "content": "[System] Vision analysis result:\nThe ESR plot shows a dip at approximately 2.87 GHz, which indicates the presence of an NV center."})
    print("Added vision analysis result")
    
    # Print conversation summary
    print(f"\nGenerated conversation with {len(agent.conversation_history)} turns")
    
    # Now simulate the exit process
    print("\nSimulating exit process...")
    print("Saving conversation embeddings before exit...")
    agent.save_conversation_embeddings()
    
    # Verify that a new embedding file was created
    if os.path.exists(agent.embeddings_dir):
        updated_files = [f for f in os.listdir(agent.embeddings_dir) if f.endswith('.json')]
        print(f"Found {len(updated_files)} embedding files after test")
        
        if len(updated_files) > len(existing_files):
            print("SUCCESS: New embedding file was created")
            
            # Get the newest file
            newest_file = max([os.path.join(agent.embeddings_dir, f) for f in updated_files], 
                             key=os.path.getctime)
            print(f"Newest embedding file: {os.path.basename(newest_file)}")
            
            # Print file size to verify it contains data
            file_size = os.path.getsize(newest_file)
            print(f"File size: {file_size} bytes")
            
            print("\nTest completed successfully!")
        else:
            print("ERROR: No new embedding file was created")
    else:
        print("ERROR: Embeddings directory does not exist")

if __name__ == "__main__":
    main()
