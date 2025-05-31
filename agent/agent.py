import os
import json
import re
import subprocess
import traceback
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union

try:
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

from anthropic_engine import call_llm, call_vision
# from deepseek_engine import call_llm, call_vision
# from anthropic_engine import call_vision
# from deepseek_engine import call_llm
from rag_engine import embed_text, save_embeddings, load_embeddings, search_similar

class NVExperimentAgent:
    def __init__(self):
        self.project_root_dir = 'projects'
        self.project_name = 'NVExperiment'
        self.runs_dir_name = 'runs'
        self.run_dir_prefix = 'run'

        file_ts = self._current_timestamp_for_filename()
        self.run_dir = f"{self.run_dir_prefix}_{file_ts}"
        self.base_dir = os.path.join(self.project_root_dir, self.project_name, self.runs_dir_name, self.run_dir)

        self.default_dir = os.path.join(self.project_root_dir, self.project_name)
        
        self.config_dir = os.path.join(self.base_dir, "configs")
        self.data_dir   = os.path.join(self.base_dir, "data")
        self.logs_dir   = os.path.join(self.base_dir, "logs")

        self.scripts_dir = "experiment_scripts"  # updated directory for scripts
        
        # Directory for storing embeddings
        self.embeddings_dir = os.path.join(self.project_root_dir, self.project_name, "embeddings")
        os.makedirs(self.embeddings_dir, exist_ok=True)

        # The entire conversation history (user messages, assistant messages, actions, etc.)
        self.conversation_history = []
        
        # Memory system configuration
        self.memory_autosave_interval = 5  # Save every 5 turns
        self.max_memories = 200  # Maximum number of memories to keep
        self.embedding_model = None
        
        # Initialize memory system
        self._initialize_memory_system()

        # Extended system instruction with updated run command and vision option details
        self.system_instruction = (
            f"""You are the NVExperimentAgent. You maintain a full conversation history, which includes:

- All user messages,
- All assistant messages (your own),
- All actions you have taken (read/write/run/vision),
- The results of those actions.

You have the following constraints and abilities:

1) Chain-of-Thought & Confidentiality:
   - You must produce exactly one `<think>` block per response, containing your private chain-of-thought.
   - Do not reveal this chain-of-thought to the user except within the `<think> … </think>` block (which the system may hide).

2) Actions:
   - You may produce zero or more `<action>` blocks, each containing valid JSON.
   - The `<action>` block must have the form:
     ```
     <action>
     {{
       "type": "...",
       "content": ...
     }}
     </action>
     ```
   - The `"type"` must be one of: `"message"`, `"read"`, `"write"`, `"run"`, or `"vision"`.

3) Memory Management:
   The agent maintains two types of memories:

   a) Error Memories:
      - Automatically saved when errors occur during experiments
      - Contains error type, severity, message, and experiment context
      - Used to track recurring issues and their resolutions
      - Stored in: {self.embeddings_dir}\errors\

   b) General Memories:
      - Stores recent conversation context and experiment metadata
      - Includes experiment type, phase, and analyzed plots
      - Helps maintain context across experiment sessions
      - Stored in: {self.embeddings_dir}\general\

   Helper Functions:
   - _get_current_experiment_type(): Determines experiment type from recent commands
   - _get_current_phase(): Identifies current phase (initialization, calibration, measurement, etc.)
   - save_conversation_embeddings(): Saves both error and general memories with metadata

In order to execute the script, you may use one of two cases. The first case is the default case, where there aren't any specific configs that the user wishes to change and you may simply read from the default base directories. In that case, follow the below instructions:
   
3) Security & Directory Rules:
   - Read Access: Only from the `configs\\` or `data\\` directories.
   - Write Access: Only to the `configs\\` or `data\\` directories.
   - Run Access: Only scripts in the `scripts\\` directory.
   - For `write`, `run`, or `vision` actions, always ask user permission first. If the user says “no,” do not proceed.
   
4) Key File Paths & Self.base_dir:
   - All outputs, file paths, or results must be written to the directory {self.base_dir}.
   - Default case (when no new config file is specified): Use the following default file paths:
     - `default_esr_config`: `{self.default_dir}\\configsdefault_esr_config.json`
     - `default_find_nv_config`: `{self.default_dir}\\configs\\default_find_nv_config.json`
     - `default_galvo_scan_config`: `{self.default_dir}\\configs\\default_galvo_scan_config.json`
     - `default_optimize_config`: `{self.default_dir}\\configs\\default_optimize_config.json`
   
5) Run Command Options:
   - The run command must include one of the following four options: ESR, find_nv, galvo_scan, or optimize.
   - IMPORTANT: You MUST include the --output-dir parameter in your command to specify where results should be saved.
   - Always use the current run's data directory as the output directory: projects\\NVExperiment\\runs\\run_(insert TIMESTAMP here)\\data\\
   - The complete command format should be:
         py projects\\experiment_scripts\\<script_name>.py --config <config_file> --output-dir projects\\NVExperiment\\runs\\run_(insert TIMESTAMP here)\\data\\
     where <script_name> is one of ESR, find_nv, galvo_scan, or optimize.

6) Vision Option:
   - In addition to running scripts, you can analyze plot images.
   - Use the command: `vision <plot_file_path>`.
   - The plot file must reside in the `data\\` directory.
   - Expected plots and their paths:
     - `{self.base_dir}\\data\\ESR_plot.png`
     - `{self.base_dir}\\data\\FindNV_plot.png`
     - `{self.base_dir}\\data\\GalvoScan_plot.png`
     - `{self.base_dir}\\data\\Optimization_plot.png`
   - For `GalvoScan_plot.png`, NVs are associated with large bright dots; estimate and read out the center coordinates of bright dots for subsequent steps.

7) Usage Flow:
   - Initial Analysis: Begin by reading the output from the most recent experiment (if experiments have been run) stored in the `data\\` directory. Analyze these results for insights.
   - Reflection & Adjustment: Reflect on the insights gained and decide on adjustments for the next run.
   - Configuration Reading: 
     - Default Case: Read the default configuration from the appropriate file (e.g., `{self.default_dir}\\configs\\default_esr_config.json`).
     - Non-default Case: Read the configuration from the new file path provided by the user.
   - Configuration Writing: 
     - Based on the reflection, write a new or updated configuration.
     - Default Case: Write to a new file under {self.base_dir} using default directory paths if no custom file is specified.
     - Non-default Case: Write to the user-specified configuration file path.
   - Experiment Execution: Run the desired experiment with:
     ```
     py {self.default_dir}\\scripts\\<script_name>.py --config <config_file> --output-dir projects\\NVExperiment\\runs\\run_(insert TIMESTAMP here)\\data\\
     ```
     where `<script_name>` is one of: `ESR`, `find_nv`, `galvo_scan`, or `optimize`.

8) Behavior & Permissions:
   - When you `<read>` a file, you receive its content internally. If you want the user to see it, produce an `<action type="message">` block.
   - When you `<write>` a file, ask the user permission. If denied, do not write.
   - When you `<run>` or `<vision>` a command, ask the user permission. If denied, do not proceed.
   - Use `<action type="message">` to communicate with the user.

9) Output Format:
   - The response must have exactly one `<think>` block and then zero or more `<action>` blocks.
   - Example Minimal Structure:
     ```
     <think>I will read the default configuration file or the user-specified configuration based on the provided case.</think>
     <action>
     {{
       "type": "read",
       "content": "{self.default_dir}\\configs\\default_esr_config.json"
     }}
     </action>
     <action>
     {{
       "type": "write",
       "content": {{
         "path": f"{self.base_dir}\\configs\\my_new_experiment_config.json",
         "data": "<updated configuration dictionary>"
       }}
     }}
     </action>
     ```
   - Always ensure that file operations and outputs are associated with {self.base_dir}.

10) Non-Default vs. Default Case Summary:
    - Default Case:  
      - No new config file is provided by the user.
      - Use the default configuration files located in the `{self.default_dir}\\configs\\` directory.
      - New outputs and any created files should be within {self.base_dir}.
    - Non-Default Case:  
      - The user requests updates to the config file.
      - You should read and then generate a modified configuration to {self.base_dir}\\configs\\.
      - All outputs are still directed to {self.base_dir}, but the config file operations occur at the new path within {self.base_dir}.

11) Restrictions:
    - Do not reveal or replicate your chain-of-thought except inside the `<think>` block.
    - Do not produce any actions outside of `"message"`, `"read"`, `"write"`, `"run"`, or `"vision"`.

"""
        )

        os.makedirs(self.logs_dir, exist_ok=True)
        # Get a file-safe timestamp
        file_ts = self._current_timestamp_for_filename()
        self.logfile_path = os.path.join(self.logs_dir, f"agent_history_{file_ts}.log")
        
        # Directory for storing structured error logs
        self.error_logs_dir = os.path.join(self.logs_dir, "errors")
        os.makedirs(self.error_logs_dir, exist_ok=True)
        
        # Track current experiment errors
        self.current_errors = []
        
        # Memory management
        self.memory_dir = os.path.join(self.project_root_dir, self.project_name, "memory")
        self.error_memory_dir = os.path.join(self.memory_dir, "errors")
        self.general_memory_dir = os.path.join(self.memory_dir, "general")
        os.makedirs(self.error_memory_dir, exist_ok=True)
        os.makedirs(self.general_memory_dir, exist_ok=True)
        
        # Memory indices
        self.error_memory_index = {}
        self.general_memory_index = {}

    def _current_timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    def _current_timestamp_for_filename(self):
        # File-safe timestamp format (e.g., 20250219_101530)
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    def _log(self, role: str, content: str):
        ts = self._current_timestamp()
        line = f"[{ts}] {role.upper()}: {content}\n"
        with open(self.logfile_path, "a", encoding="utf-8") as f:
            f.write(line)
            
    # ===== Memory System Methods =====
    
    def _initialize_memory_system(self):
        """Initialize the memory system and load existing memories."""
        # Ensure memory directories exist
        for mem_type in ['error', 'general']:
            mem_dir = os.path.join(self.embeddings_dir, f"{mem_type}s")
            os.makedirs(mem_dir, exist_ok=True)
        
        # Load existing memory indices
        self._load_memory_indices()
        
        # Initialize RAG components if available
        self._initialize_rag_system()
    
    def _initialize_rag_system(self):
        """Initialize the RAG system with necessary components."""
        if RAG_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("RAG system initialized with all-MiniLM-L6-v2 model")
            except Exception as e:
                print(f"Warning: Could not initialize RAG system: {e}")
    
    def _load_memory_indices(self):
        """Load memory indices from disk."""
        for mem_type in ['error', 'general']:
            index_file = os.path.join(self.embeddings_dir, f"{mem_type}_index.json")
            if os.path.exists(index_file):
                try:
                    with open(index_file, 'r') as f:
                        setattr(self, f"{mem_type}_memory_index", json.load(f))
                except json.JSONDecodeError:
                    print(f"Warning: Could not load {index_file}, starting with empty index")
                    setattr(self, f"{mem_type}_memory_index", {})
            else:
                setattr(self, f"{mem_type}_memory_index", {})
    
    def _save_memory(self, memory: dict, memory_type: str = 'general') -> str:
        """
        Save a memory to disk and update the index.
        Returns the memory ID.
        """
        timestamp = datetime.now().isoformat()
        memory_id = f"{memory_type}_{hash(json.dumps(memory, sort_keys=True))}"
        
        # Add metadata
        memory.update({
            'id': memory_id,
            'timestamp': timestamp,
            'type': memory_type
        })
        
        # Generate embedding if content is text and model is available
        if 'content' in memory and self.embedding_model and isinstance(memory['content'], str):
            memory['embedding'] = self.get_embedding(memory['content'])
        
        # Save to file
        mem_dir = os.path.join(self.embeddings_dir, f"{memory_type}s")
        os.makedirs(mem_dir, exist_ok=True)
        mem_path = os.path.join(mem_dir, f"{memory_id}.json")
        
        with open(mem_path, 'w') as f:
            json.dump(memory, f, indent=2, default=str)
        
        # Update in-memory index
        index = getattr(self, f"{memory_type}_memory_index", {})
        index[memory_id] = {
            'path': mem_path,
            'timestamp': timestamp,
            'type': memory_type
        }
        setattr(self, f"{memory_type}_memory_index", index)
        
        # Save index to disk
        self._save_memory_index(memory_type)
        
        return memory_id
    
    def _save_memory_index(self, memory_type: str):
        """Save the memory index to disk."""
        index = getattr(self, f"{memory_type}_memory_index", {})
        index_file = os.path.join(self.embeddings_dir, f"{memory_type}_index.json")
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2, default=str)
    
    def get_embedding(self, text: str) -> list:
        """Get embedding for a piece of text."""
        if not text or not self.embedding_model:
            return []
        return self.embedding_model.encode(text).tolist()
    
    def search_memories(self, query: str, memory_type: str = None, top_k: int = 3) -> list:
        """
        Search through memories using semantic similarity.
        Returns list of (score, memory) tuples sorted by relevance.
        """
        if not query or not self.embedding_model:
            return []

        # Get query embedding
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []

        results = []
        
        # Determine which memory types to search
        search_types = [memory_type] if memory_type else ['error', 'general']
        
        for m_type in search_types:
            mem_dir = os.path.join(self.embeddings_dir, f"{m_type}s")
            if not os.path.exists(mem_dir):
                continue
                
            # Search through all memory files
            for mem_file in os.listdir(mem_dir):
                if not mem_file.endswith('.json'):
                    continue
                    
                mem_path = os.path.join(mem_dir, mem_file)
                try:
                    with open(mem_path, 'r') as f:
                        memory = json.load(f)
                        
                    # Calculate similarity if memory has an embedding
                    if 'embedding' in memory:
                        similarity = self._cosine_similarity(query_embedding, memory['embedding'])
                        results.append((similarity, memory))
                        
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error loading memory {mem_file}: {e}")
        
        # Sort by similarity score (highest first) and return top_k
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]
    
    def _cosine_similarity(self, vec_a: list, vec_b: list) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        a = np.array(vec_a)
        b = np.array(vec_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def update_conversation_memory(self, message: str, role: str = 'user'):
        """
        Update conversation memory with a new message.
        Automatically saves to memory at regular intervals.
        """
        # Add to conversation history
        self.conversation_history.append({
            'role': role,
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Auto-save memories at regular intervals
        if len(self.conversation_history) % self.memory_autosave_interval == 0:
            self._auto_save_memories()
    
    def _auto_save_memories(self):
        """Automatically save important memories from recent conversation."""
        if not self.conversation_history:
            return
            
        # Get the most recent messages
        recent_messages = self.conversation_history[-self.memory_autosave_interval:]
        
        # Create a memory of the conversation chunk
        memory = {
            'type': 'conversation_chunk',
            'content': recent_messages,
            'timestamp': datetime.now().isoformat(),
            'embedding': self.get_embedding('\n'.join([str(m.get('content', '')) for m in recent_messages]))
        }
        
        # Save the memory
        self._save_memory(memory, 'general')
        
        # Prune old memories if we have too many
        self._prune_old_memories()
    
    def _prune_old_memories(self):
        """Remove oldest memories when exceeding max_memories."""
        for mem_type in ['error', 'general']:
            mem_dir = os.path.join(self.embeddings_dir, f"{mem_type}s")
            if not os.path.exists(mem_dir):
                continue
                
            # Get all memory files sorted by modification time (oldest first)
            try:
                mem_files = sorted(
                    (os.path.join(mem_dir, f) for f in os.listdir(mem_dir) if f.endswith('.json')),
                    key=os.path.getmtime
                )
                
                # If we're over the limit, remove the oldest ones
                while len(mem_files) > self.max_memories // 2:  # Split max between error and general
                    os.remove(mem_files.pop(0))
                    
            except OSError as e:
                print(f"Error pruning {mem_type} memories: {e}")

    def _parse_think(self, text: str) -> str:
        """
        Extract content from <think>...</think> block.
        """
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _build_prompt(self) -> str:
        """
        Combine system_instruction, RAG results, and conversation_history into a single prompt.
        Also include suggestions for relevant plots.
        """
        prompt = self.system_instruction.strip()
        
        # Get the most recent user message for RAG query
        recent_user_messages = [turn["content"] for turn in self.conversation_history 
                               if turn["role"] == "user"]
        
        if recent_user_messages:
            # Use the most recent user message as the query
            query = recent_user_messages[-1]
            
            # RX 05142025
            # Log that we're performing a RAG query
            print(f"[RAG] Performing RAG query for: '{query[:50]}...' if len(query) > 50 else query")
            
            # Perform RAG search
            relevant_contexts = self._get_rag_context(query)
            
            def _get_rag_context(self, query: str) -> str:
                """Get relevant context using specialized RAG queries"""
                # First check if this is an error-related query
                error_terms = ['error', 'fail', 'issue', 'problem', 'warning', 'exception', 'crash']
                is_error_query = any(term in query.lower() for term in error_terms)
                
                context_parts = []
                
                # If it's an error query or there are current errors, search error memories first
                if is_error_query or self.current_errors:
                    error_memories = self._search_memory(query, memory_type='error')
                    if error_memories:
                        context_parts.append("Relevant error context:")
                        for memory in error_memories:
                            # Add error information
                            for error in memory.get('errors', []):
                                context_parts.append(f"- {error['error_type']} error in {error['experiment']}: {error['message']}")
                                if error['context']:
                                    context_parts.append(f"  Context: {error['context']}")
                
                # Always search general memories
                general_memories = self._search_memory(query, memory_type='general')
                if general_memories:
                    context_parts.append("\nRelevant general context:")
                    for memory in general_memories:
                        # Add conversation snippets
                        for turn in memory.get('conversation', []):
                            context_parts.append(f"- {turn['role']}: {turn['content']}")
                        # Add metadata
                        metadata = memory.get('metadata', {})
                        if metadata:
                            context_parts.append(f"  (During {metadata['experiment']} experiment, {metadata['phase']} phase)")
                
                # Combine all context
                if context_parts:
                    return "\n".join(context_parts)
                return ""
            
            if relevant_contexts:
                print(f"[RAG] Retrieved relevant context from previous conversations")
                print(f"Retrieved context: {relevant_contexts}")
                prompt += "\n\nRelevant context from previous conversations:\n"
                prompt += relevant_contexts
            else:
                print("[RAG] No relevant context found in previous conversations")
            
            # Check for relevant plots
            relevant_plots = self._get_relevant_plots(query)
            if relevant_plots:
                plot_filenames = [os.path.basename(p) for p in relevant_plots]
                print(f"[RAG] Found relevant plots: {plot_filenames}")
                self._log("rag", f"Relevant plots: {plot_filenames}")
                
                prompt += "\n\nRelevant plots that might help with this query:\n"
                for plot_path in relevant_plots:
                    plot_filename = os.path.basename(plot_path)
                    prompt += f"- {plot_filename}\n"
                prompt += "\nYou can analyze these plots using the 'vision' action if needed."
            else:
                print("[RAG] No relevant plots found")
        
        # Add conversation history
        for turn in self.conversation_history:
            role = turn["role"]
            content = turn["content"]
            if role == "user":
                prompt += f"\nUser: {content}"
            elif role == "assistant":
                prompt += f"\nAssistant: {content}"
            else:
                prompt += f"\n{role.capitalize()}: {content}"
        
        prompt += "\n\nPlease respond with a <think> block and any <action> blocks you need for the next step. Please carefully wait for user and experiment feedback before proceeding to too many actions."
        return prompt

    def ask_human_for_permission(self, description: str) -> bool:
        #RX 05142025
        """
        Ask the user on the console for permission and log the response.
        """
        self._log("action", f"(ASK PERMISSION) {description}")
        self.conversation_history.append({
            "role": "assistant",
            "content": f"Agent requests permission to: {description}"
        })
        print(f"[System] Agent requests permission to: {description}")
        ans = input("Grant permission? (yes/no): ").strip().lower()
        self._log("user", f"(permission) {ans}")
        self.conversation_history.append({
            "role": "user",
            "content": f"(permission) {ans}"
        })
        ans = "yes"
        return (ans == "yes")

    def handle_user_input(self, user_message: str):
        """
        Process user prompt: log it, build the prompt, call the LLM, parse and execute actions.
        """
        print(f"\n[Agent] Processing user input: '{user_message[:50]}{'...' if len(user_message) > 50 else ''}'")  
        self._log("user", user_message)
        
        # Update conversation memory with the new user message
        self.update_conversation_memory(user_message, 'user')
        
        # Search for relevant memories to include in the context
        relevant_memories = self.search_memories(user_message, top_k=3)
        if relevant_memories:
            print("\n[Memory] Found relevant context from previous interactions:")
            for score, memory in relevant_memories:
                content = memory.get('content', '')
                if isinstance(content, list):
                    content = '\n'.join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in content if 'content' in m])
                print(f"- ({score:.2f}) {content[:100]}..." if len(str(content)) > 100 else f"- ({score:.2f}) {content}")
        
        self._log("agent", "Building prompt with RAG context")
        full_prompt = self._build_prompt()
        
        print("[Agent] Calling LLM with enhanced prompt...")
        llm_response = call_llm(
            user_prompt=full_prompt,
            system_message=self.system_instruction,
            #model = "deepseek-chat",
            max_tokens=3000,
            temperature=0.7
        )
        self._log("assistant", llm_response)
        self.conversation_history.append({"role": "assistant", "content": llm_response})
        chain_of_thought = self._parse_think(llm_response)
        if chain_of_thought:
            self._log("assistant", f"(THINK) {chain_of_thought}")
            self.conversation_history.append({"role": "assistant", "content": f"(THINK) {chain_of_thought}"})

        actions = self._parse_actions(llm_response)

        for action_dict in actions:
            a_type = action_dict.get("type", "").lower()
            content = action_dict.get("content", "")

            if a_type == "message":
                self._action_message(content)
            elif a_type == "read":
                self._action_read_file(content)
            elif a_type == "write":
                if self.ask_human_for_permission(f"Write file: {content}"):
                    self._action_write_file(content)
                else:
                    print("[System] Write denied by user.")
                    self._log("action", f"WRITE DENIED for {content}")
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": f"[Agent] WRITE DENIED for {content}"
                    })
            elif a_type == "run":
                if self.ask_human_for_permission(f"Run command: {content}"):
                    self._action_run_command(content)
                else:
                    print("[System] Run denied by user.")
                    self._log("action", f"RUN DENIED for {content}")
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": f"[Agent] RUN DENIED for {content}"
                    })
            elif a_type == "vision":
                if self.ask_human_for_permission(f"Analyze plot: {content}"):
                    self._action_vision(content)
                else:
                    print("[System] Vision analysis denied by user.")
                    self._log("action", f"VISION DENIED for {content}")
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": f"[Agent] VISION DENIED for {content}"
                    })
            else:
                print(f"[System] Unknown action type: {a_type}")
                self._log("action", f"Unknown action {a_type}")
                self.conversation_history.append({
                    "role": "assistant",
                    "content": f"[Agent] Unknown action {a_type}"
                })

    def _parse_actions(self, llm_text: str):
        """
        Return a list of JSON objects found in <action>...</action> blocks.
        """
        pattern = r"<action>(.*?)</action>"
        matches = re.findall(pattern, llm_text, flags=re.DOTALL)
        actions = []
        for m in matches:
            try:
                a_dict = json.loads(m.strip())
                actions.append(a_dict)
            except json.JSONDecodeError:
                pass
        return actions

    def _action_message(self, message_content: str):
        """
        Print a message and log it.
        """
        print(message_content)
        self._log("action", f"MESSAGE: {message_content}")
        self.conversation_history.append({"role": "assistant", "content": message_content})

    def _action_read_file(self, filepath: str):
        """
        Read a file from allowed directories (configs\\ or data\\).
        """
        self._log("action", f"READ: {filepath}")
        allowed_prefixes = [self.config_dir, os.path.join(self.default_dir, 'configs'), self.data_dir]

        if not any(filepath.startswith(p) for p in allowed_prefixes):
            msg = f"[System] READ denied: {filepath} is not in allowed directories."
            print(msg)
            self._log("action", msg)
            self.conversation_history.append({"role": "assistant", "content": msg})
            return
        if not os.path.exists(filepath):
            msg = f"[System] File not found: {filepath}"
            print(msg)
            self._log("action", msg)
            self.conversation_history.append({"role": "assistant", "content": msg})
            return
        with open(filepath, 'r', encoding="utf-8") as f:
            content = f.read()
        self.conversation_history.append({
            "role": "assistant",
            "content": f"(Read file) {filepath} with content:\n{content}"
        })

    def _action_write_file(self, content: dict):
        """
        Write JSON data to a file in allowed directories (configs\\ or data\\).
        """
        self._log("action", f"WRITE file with content: {json.dumps(content, indent=2)}")
        if not isinstance(content, dict):
            msg = "[System] Write error: content is not a dict."
            print(msg)
            self._log("action", msg)
            self.conversation_history.append({"role": "assistant", "content": msg})
            return
        filepath = content.get("path", "")
        filedata = content.get("data", None)
        if not filepath or filedata is None:
            msg = "[System] Write error: missing 'filepath' or 'data'."
            print(msg)
            self._log("action", msg)
            self.conversation_history.append({"role": "assistant", "content": msg})
            return
        allowed_prefixes = [self.config_dir, self.data_dir]
        if not any(filepath.startswith(p) for p in allowed_prefixes):
            msg = f"[System] WRITE denied: {filepath} is not in allowed directories."
            print(msg)
            self._log("action", msg)
            self.conversation_history.append({"role": "assistant", "content": msg})
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(filedata, f, indent=2)
        msg = f"[System] Wrote file: {filepath}"
        print(msg)
        self._log("action", msg)
        self.conversation_history.append({"role": "assistant", "content": msg})

    def _parse_run_command(self, command: str) -> dict:
        """
        Parse the run command to extract the script and config file.
        Expected format:
            py experiment_scripts\\<script_name>.py --config <config_file> [--output-dir <output_directory>]
        where <script_name> is one of ESR, find_nv, galvo_scan, or optimize.
        """
        # More flexible pattern to handle different path formats and whitespace variations
        pattern = r"py\s+(?P<path>(?:projects[\\/])?(?:experiment_scripts|NVExperiment[\\/]scripts)[\\/](ESR\.py|find_nv\.py|galvo_scan\.py|optimize\.py))\s+--config\s+(?P<config>[\w\\./-]+)(?:\s+--output-dir\s+(?P<output_dir>[\w\\./-]+))?"
        m = re.search(pattern, command)
        if m:
            result = {"script": m.group("path"), "config": m.group("config")}
            if m.group("output_dir"):
                result["output_dir"] = m.group("output_dir")
            return result
        else:
            # More informative error message that includes the command that failed to parse
            allowed_scripts = "ESR.py, find_nv.py, galvo_scan.py, or optimize.py"
            error_msg = f"Run command parsing error for command: '{command}'. \nCommand must be in the format: py experiment_scripts/<script_name>.py --config <config_file> [--output-dir <output_directory>] \nwhere <script_name> is one of {allowed_scripts}"
            raise ValueError(error_msg)


    def _get_current_experiment_type(self) -> str:
        """Get the type of experiment currently being run"""
        for turn in reversed(self.conversation_history):
            content = turn.get('content', '')
            if 'RUN:' in content:
                if 'ESR.py' in content:
                    return 'ESR'
                elif 'find_nv.py' in content:
                    return 'find_nv'
                elif 'galvo_scan.py' in content:
                    return 'galvo_scan'
                elif 'optimize.py' in content:
                    return 'optimize'
        return 'unknown'

    def _get_current_phase(self) -> str:
        """Get the current phase of the experiment"""
        # Look at last 5 turns to determine phase
        recent_turns = self.conversation_history[-5:]
        for turn in reversed(recent_turns):
            content = turn.get('content', '').lower()
            
            # Check for phase indicators
            if any(term in content for term in ['initializing', 'setup', 'starting']):
                return 'initialization'
            elif any(term in content for term in ['calibrating', 'tuning']):
                return 'calibration'
            elif any(term in content for term in ['measuring', 'collecting data']):
                return 'measurement'
            elif any(term in content for term in ['analyzing', 'processing']):
                return 'analysis'
            elif any(term in content for term in ['error', 'failed', 'issue']):
                return 'error_handling'
        return 'unknown'

    def save_conversation_embeddings(self):
        """Save the current conversation with separate error and general memories"""
        # Create directories
        error_dir = os.path.join(self.embeddings_dir, 'errors')
        general_dir = os.path.join(self.embeddings_dir, 'general')
        os.makedirs(error_dir, exist_ok=True)
        os.makedirs(general_dir, exist_ok=True)
        
        timestamp = self._current_timestamp_for_filename()
        
        # Save error memories if there are errors
        if self.current_errors:
            error_data = {
                'timestamp': self._current_timestamp(),
                'errors': [
                    {
                        'error_type': error.error_type,
                        'severity': error.severity,
                        'experiment': error.experiment,
                        'message': error.message
                    }
                    for error in self.current_errors
                ],
                'metadata': {
                    'experiment_type': self._get_current_experiment_type(),
                    'experiment_phase': self._get_current_phase()
                }
            }
            
            error_file = os.path.join(error_dir, f"error_{timestamp}.json")
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, default=str)
        
        # Save general memory
        general_data = {
            'timestamp': self._current_timestamp(),
            'conversation': self.conversation_history[-3:],  # Last 3 turns
            'metadata': {
                'experiment_type': self._get_current_experiment_type(),
                'experiment_phase': self._get_current_phase(),
                'analyzed_plots': self._track_analyzed_plots()
            }
        }
        
        general_file = os.path.join(general_dir, f"general_{timestamp}.json")
        with open(general_file, 'w', encoding='utf-8') as f:
            json.dump(general_data, f, indent=2, default=str)

    def _action_run_command(self, command: str):
        """
        Run a shell command (only if in 'scripts\' directory and is one of the allowed scripts), capturing output.
        """
        self._log("action", f"RUN: {command}")
        self._log("action", f"RUN: {command}")

@dataclass
class ExperimentError:
    """Structured representation of an experiment error"""
    error_id: str
    timestamp: datetime
    error_type: str  # 'hardware', 'config', 'data_quality', 'system'
    severity: str    # 'critical', 'warning', 'info'
    experiment: str  # 'ESR', 'find_nv', 'galvo_scan', 'optimize'
    message: str
    stderr: Optional[str] = None
    stdout: Optional[str] = None
    context: Dict[str, Any] = None

def _categorize_error(self, error_message: str, stderr: Optional[str] = None) -> Dict[str, str]:
    """Categorize error type and severity based on error message"""
    # Determine error type
    if any(hw_term in (error_message + (stderr or '')).lower() 
           for hw_term in ['timeout', 'connection', 'device', 'hardware']):
        error_type = 'hardware'
    elif 'config' in (error_message + (stderr or '')).lower():
        error_type = 'config'
    elif any(data_term in (error_message + (stderr or '')).lower() 
             for data_term in ['signal', 'noise', 'quality', 'threshold']):
        error_type = 'data_quality'
    else:
        error_type = 'system'
    
    # Determine severity
    if any(critical_term in (error_message + (stderr or '')).lower() 
           for critical_term in ['fail', 'error', 'exception', 'crash']):
        severity = 'critical'
    elif any(warning_term in (error_message + (stderr or '')).lower() 
             for warning_term in ['warning', 'low', 'weak']):
        severity = 'warning'
    else:
        severity = 'info'
    
    return {'error_type': error_type, 'severity': severity}

def _gather_error_context(self) -> Dict[str, Any]:
    """Gather context information when an error occurs"""
    context = {}
    
    # Get current config
    try:
        config_file = self._get_current_config()
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                context['config_used'] = json.load(f)
    except Exception:
        context['config_used'] = None
    
    # Get available plots
    context['plot_refs'] = self._get_available_plots()
    
    # Get experiment results if available
    try:
        results_file = os.path.join(self.data_dir, 
                                   f"{self._get_current_experiment()}_results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                context['results'] = json.load(f)
    except Exception:
        context['results'] = None
    
    return context

def _log_experiment_error(self, error: ExperimentError):
    """Log a structured experiment error"""
    # Add to current errors list
    self.current_errors.append(error)
    
    # Save to error log file
    error_file = os.path.join(self.error_logs_dir, 
                             f"error_{error.error_id}_{self._current_timestamp_for_filename()}.json")
    
    error_data = {
        'error_id': error.error_id,
        'timestamp': error.timestamp.isoformat(),
        'error_type': error.error_type,
        'severity': error.severity,
        'experiment': error.experiment,
        'message': error.message,
        'stderr': error.stderr,
        'stdout': error.stdout,
        'context': error.context
    }
    
    with open(error_file, 'w') as f:
        json.dump(error_data, f, indent=2)

    def _action_run_command(self, command: str):
        """Run a shell command with structured error handling"""
        try:
            parsed = self._parse_run_command(command)
            allowed_scripts = [f"{self.default_dir}\\scripts\\ESR.py", 
                            f"{self.default_dir}\\scripts\\find_nv.py", 
                            f"{self.default_dir}\\scripts\\galvo_scan.py", 
                            f"{self.default_dir}\\scripts\\optimize.py"]
            script_normalized = parsed["script"].replace("/", "\\")

            if script_normalized not in allowed_scripts:
                raise ValueError("Command not allowed: script not among allowed options.")
            
            # Ensure the data directory exists
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Log the command
            self._log("action", f"Running command: {command}")
            
            # Create command memory
            command_memory = {
                'type': 'command_execution',
                'command': command,
                'script': script_normalized,
                'config': parsed.get('config'),
                'output_dir': parsed.get('output_dir', self.data_dir),
                'timestamp': datetime.now().isoformat(),
                'status': 'started',
                'experiment_type': self._get_current_experiment_type(),
                'phase': self._get_current_phase()
            }
            self._save_memory(command_memory, 'general')
            
            # Run the command
            result = subprocess.run(command, shell=True, check=True, capture_output=True)
            stdout_text = result.stdout.decode()
            stderr_text = result.stderr.decode()
            
            # Update command memory with results
            command_memory.update({
                'status': 'completed',
                'return_code': 0,
                'stdout': stdout_text,
                'stderr': stderr_text,
                'completed_at': datetime.now().isoformat()
            })
            self._save_memory(command_memory, 'general')
            
            # Check for warnings in stderr
            if stderr_text:
                # Categorize as warning if stderr exists but command didn't fail
                error_info = self._categorize_error(stderr_text)
                if error_info['severity'] == 'critical':
                    error_info['severity'] = 'warning'  # Downgrade since command succeeded
                
                error = ExperimentError(
                    error_id=f"warn_{self._current_timestamp_for_filename()}",
                    timestamp=datetime.now(),
                    error_type=error_info['error_type'],
                    severity=error_info['severity'],
                    experiment=self._get_current_experiment(),
                    message=stderr_text,
                    stderr=stderr_text,
                    stdout=stdout_text,
                    context=self._gather_error_context()
                )
                self._log_experiment_error(error)
                
                # Save warning to memory
                warning_memory = {
                    'type': 'command_warning',
                    'command': command,
                    'error_type': error_info['error_type'],
                    'severity': error_info['severity'],
                    'message': stderr_text,
                    'timestamp': datetime.now().isoformat(),
                    'experiment_type': self._get_current_experiment_type(),
                    'phase': self._get_current_phase()
                }
                self._save_memory(warning_memory, 'error')
            
            # Format output message
            out_msg = "[System] Command output:\n" + stdout_text
            if stderr_text:
                out_msg += "\n[System] Command warnings:\n" + stderr_text
            
            print(out_msg)
            self._log("action", f"RUN OUTPUT: {stdout_text}")
            self.conversation_history.append({"role": "assistant", "content": out_msg})
            self.update_conversation_memory(out_msg, 'assistant')
            
            return out_msg
            
        except Exception as e:
            # Handle critical errors
            error_message = str(e)
            error_info = self._categorize_error(error_message)
            stderr = getattr(e, 'stderr', '')
            if hasattr(e, 'stderr') and hasattr(e.stderr, 'decode'):
                stderr = e.stderr.decode()
            stdout = getattr(e, 'stdout', '')
            if hasattr(e, 'stdout') and hasattr(e.stdout, 'decode'):
                stdout = e.stdout.decode()
            
            # Save error to memory
            error_memory = {
                'type': 'command_error',
                'command': command,
                'error_type': error_info['error_type'],
                'severity': 'critical',
                'message': error_message,
                'traceback': traceback.format_exc(),
                'stderr': stderr,
                'stdout': stdout,
                'timestamp': datetime.now().isoformat(),
                'experiment_type': self._get_current_experiment_type(),
                'phase': self._get_current_phase()
            }
            self._save_memory(error_memory, 'error')
            
            error = ExperimentError(
                error_id=f"err_{self._current_timestamp_for_filename()}",
                timestamp=datetime.now(),
                error_type=error_info['error_type'],
                severity='critical',
                experiment=self._get_current_experiment(),
                message=error_message,
                stderr=stderr,
                stdout=stdout,
                context=self._gather_error_context()
            )
            
            self._log_experiment_error(error)
            
            err_msg = f"[System] Error running command: {error_message}"
            print(err_msg)
            self._log("action", err_msg)
            self.conversation_history.append({"role": "assistant", "content": err_msg})
            self.update_conversation_memory(err_msg, 'assistant')
            
            raise

    def _get_current_experiment_type(self) -> str:
        """Get the type of experiment currently being run"""
        for turn in reversed(self.conversation_history):
            content = turn.get('content', '')
            if 'RUN:' in content:
                if 'ESR.py' in content:
                    return 'ESR'
                elif 'find_nv.py' in content:
                    return 'find_nv'
                elif 'galvo_scan.py' in content:
                    return 'galvo_scan'
                elif 'optimize.py' in content:
                    return 'optimize'
        return 'unknown'

    def _get_current_phase(self) -> str:
        """Get the current phase of the experiment"""
        # Look at last 5 turns to determine phase
        recent_turns = self.conversation_history[-5:]
        for turn in reversed(recent_turns):
            content = turn.get('content', '').lower()
            
            # Check for phase indicators
            if any(term in content for term in ['initializing', 'setup', 'starting']):
                return 'initialization'
            elif any(term in content for term in ['calibrating', 'tuning']):
                return 'calibration'
            elif any(term in content for term in ['measuring', 'collecting data']):
                return 'measurement'
            elif any(term in content for term in ['analyzing', 'processing']):
                return 'analysis'
            elif any(term in content for term in ['error', 'failed', 'issue']):
                return 'error_handling'
        
        return 'unknown'

    def _track_analyzed_plots(self) -> List[str]:
        """Track which plots have been analyzed in the current session"""
        analyzed_plots = []
        for turn in self.conversation_history:
            content = turn.get('content', '')
            if 'VISION:' in content:
                plot_file = content.split('VISION:')[1].strip()
                analyzed_plots.append(os.path.basename(plot_file))
        return analyzed_plots

    def _update_memory_indices(self):
        """Update memory indices for both error and general memories"""
        # Update error memory index
        if self.current_errors:
            error_key = f"error_{self._current_timestamp_for_filename()}"
            self.error_memory_index[error_key] = {
                'timestamp': self._current_timestamp(),
                'error_types': [error.error_type for error in self.current_errors],
                'experiment': self._get_current_experiment_type(),
                'phase': self._get_current_phase()
            }
            
            # Save error memory
            error_memory_path = os.path.join(self.error_memory_dir, f"{error_key}.json")
            with open(error_memory_path, 'w') as f:
                json.dump({
                    'errors': [{
                        'error_type': error.error_type,
                        'severity': error.severity,
                        'experiment': error.experiment,
                        'message': error.message,
                        'context': error.context
                    } for error in self.current_errors],
                    'metadata': self.error_memory_index[error_key]
                }, f, indent=2)
        
        # Update general memory index
        general_key = f"general_{self._current_timestamp_for_filename()}"
        self.general_memory_index[general_key] = {
            'timestamp': self._current_timestamp(),
            'experiment': self._get_current_experiment_type(),
            'phase': self._get_current_phase(),
            'plots': self._track_analyzed_plots(),
            'commands': [turn['content'] for turn in reversed(self.conversation_history)
                        if 'RUN:' in turn.get('content', '')][:3]
        }
        
        # Save general memory
        general_memory_path = os.path.join(self.general_memory_dir, f"{general_key}.json")
        with open(general_memory_path, 'w') as f:
            json.dump({
                'conversation': [{
                    'role': turn['role'],
                    'content': turn['content'],
                    'timestamp': self._current_timestamp()
                } for turn in self.conversation_history[-3:]],
                'metadata': self.general_memory_index[general_key]
            }, f, indent=2)

    def _search_memory(self, query: str, memory_type: str = 'general') -> List[Dict]:
        """Search through memory indices to find relevant context"""
        if memory_type not in ['general', 'error']:
            raise ValueError("memory_type must be 'general' or 'error'")
            
        memory_dir = self.general_memory_dir if memory_type == 'general' else self.error_memory_dir
        memory_index = self.general_memory_index if memory_type == 'general' else self.error_memory_index
        
        relevant_memories = []
        current_exp = self._get_current_experiment_type()
        current_phase = self._get_current_phase()
        
        for key, metadata in memory_index.items():
            score = 0
            # Prioritize memories from same experiment type
            if metadata['experiment'] == current_exp:
                score += 2
            # Prioritize memories from same phase
            if metadata['phase'] == current_phase:
                score += 1
            # For error memories, check error types
            if memory_type == 'error' and any(error_type in query.lower() 
                                            for error_type in metadata.get('error_types', [])):
                score += 3
            # For general memories, check if query terms match plot names or commands
            elif memory_type == 'general':
                if any(plot.lower() in query.lower() for plot in metadata.get('plots', [])):
                    score += 2
                if any(cmd.lower() in query.lower() for cmd in metadata.get('commands', [])):
                    score += 2
                    
            if score > 0:
                memory_path = os.path.join(memory_dir, f"{key}.json")
                with open(memory_path, 'r') as f:
                    memory_data = json.load(f)
                    memory_data['relevance_score'] = score
                    relevant_memories.append(memory_data)
        
        # Sort by relevance score
        relevant_memories.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_memories[:3]  # Return top 3 most relevant memories

    def _build_vision_context(self,num_turns=4) -> str:
        """
        Build a context string for vision analysis from recent conversation history.
        Here you might simply concatenate the last few messages or summarize them.
        """
        # For illustration, we take the last 3 messages:
        recent_turns = self.conversation_history[-num_turns:]
        context_lines = [f"{turn['role']}: {turn['content']}" for turn in recent_turns]
        return " ".join(context_lines)

    def _action_vision(self, filepath: str):
        """
        Analyze a plot image from the data\\ directory using the vision model, including conversation context.
        """
        self._log("action", f"VISION: {filepath}")
        
        # Handle both absolute paths and relative paths
        if not os.path.isabs(filepath):
            # If it's just a filename, assume it's in the current run's data directory
            if os.path.basename(filepath) == filepath:
                filepath = os.path.join(self.data_dir, filepath)
            # If it starts with 'projects/data/' or similar patterns, convert to the correct path
            elif any(filepath.startswith(prefix) for prefix in ['projects/data/', 'projects\\data\\', 'data/', 'data\\']):
                plot_filename = os.path.basename(filepath)
                filepath = os.path.join(self.data_dir, plot_filename)
            # Handle paths that might be in the format projects/NVExperiment/runs/run_*/data/
            elif 'NVExperiment/runs/' in filepath.replace('\\', '/') or 'data' in filepath:
                # Try to extract just the filename if it's a complex path
                plot_filename = os.path.basename(filepath)
                filepath = os.path.join(self.data_dir, plot_filename)
        
        # Check if the file is in the allowed data directory
        if not filepath.startswith(self.data_dir):
            msg = f"[System] VISION denied: {filepath} is not in the allowed data directory ({self.data_dir})."
            print(msg)
            self._log("action", msg)
            self.conversation_history.append({"role": "assistant", "content": msg})
            return
            
        if not os.path.exists(filepath):
            msg = f"[System] File not found: {filepath}"
            print(msg)
            self._log("action", msg)
            self.conversation_history.append({"role": "assistant", "content": msg})
            return

        vision_context = self._build_vision_context()
        analysis = call_vision(filepath, additional_context=vision_context)
        msg = f"[System] Vision analysis result:\n{analysis}"
        print(msg)
        self._log("action", msg)
        self.conversation_history.append({"role": "assistant", "content": msg})
    
    def _get_available_plots(self):
        """
        Scan the data directory for plot files and return their paths.
        
        Returns:
            List of plot file paths
        """
        plot_files = []
        if os.path.exists(self.data_dir):
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.png') and any(plot_type in filename for plot_type in 
                                                  ['ESR', 'FindNV', 'GalvoScan', 'Optimization']):
                    plot_files.append(os.path.join(self.data_dir, filename))
        return plot_files
    
    def _get_relevant_plots(self, query):
        """
        Check if there are any plots in the data directory that might be relevant to the query.
        
        Args:
            query: The user's query
            
        Returns:
            List of relevant plot paths
        """
        relevant_plots = []
        
        # Define keywords for each plot type
        plot_keywords = {
            'ESR': ['esr', 'electron spin resonance', 'frequency', 'spectrum'],
            'FindNV': ['findnv', 'find nv', 'nv center', 'diamond', 'locate'],
            'GalvoScan': ['galvoscan', 'galvo', 'scan', 'mapping', 'surface'],
            'Optimization': ['optimize', 'optimization', 'parameter', 'tuning']
        }
        
        # Check if any keywords are in the query
        query_lower = query.lower()
        matching_types = []
        for plot_type, keywords in plot_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                matching_types.append(plot_type)
        
        # Get all plots in the data directory
        available_plots = self._get_available_plots()
        
        # Filter for relevant plots
        for plot_path in available_plots:
            plot_filename = os.path.basename(plot_path)
            if any(plot_type in plot_filename for plot_type in matching_types) or not matching_types:
                relevant_plots.append(plot_path)
        
        return relevant_plots
    
    def _track_analyzed_plots(self):
        """
        Track which plots have been analyzed in the current session.
        
        Returns:
            Dictionary mapping plot filenames to their analysis status
        """
        analyzed_plots = {}
        
        # Get all plots in the data directory
        available_plots = self._get_available_plots()
        for plot_path in available_plots:
            plot_filename = os.path.basename(plot_path)
            analyzed_plots[plot_filename] = False
        
        # Check which plots have been analyzed
        for turn in self.conversation_history:
            if "VISION:" in turn.get("content", ""):
                plot_path = turn["content"].split("VISION:")[1].strip()
                plot_filename = os.path.basename(plot_path)
                if plot_filename in analyzed_plots:
                    analyzed_plots[plot_filename] = True
        
        return analyzed_plots
    
    def _suggest_unanalyzed_plots(self):
        """
        Suggest plots that haven't been analyzed yet.
        
        Returns:
            List of unanalyzed plot paths
        """
        analyzed_plots = self._track_analyzed_plots()
        unanalyzed_plots = []
        
        for plot_filename, analyzed in analyzed_plots.items():
            if not analyzed:
                unanalyzed_plots.append(os.path.join(self.data_dir, plot_filename))
        
        return unanalyzed_plots
    


    def _get_rag_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant context using specialized searches for errors and general content.
        
        Args:
            query: The user's query to search against
            top_k: Number of most relevant contexts to retrieve
            
        Returns:
            String containing the most relevant contexts
        """
        combined_context = ""
        
        # 1. Do general search first
        general_query = self._build_general_query(query)
        general_results = self._search_general_embeddings(general_query)
        if general_results:
            combined_context += "\nRelevant context from previous experiments:\n"
            combined_context += general_results
        
        # 2. Do error search only if:
        # - We have active errors
        # - Query is error-related
        # - We're in an experiment that commonly has errors
        error_query = self._build_error_query(query)
        if (self.current_errors or 
            error_query['is_error_related'] or 
            error_query['current_experiment'] in ['find_nv', 'galvo_scan']):  # These often have errors
            
            error_results = self._search_error_embeddings(error_query)
            if error_results:
                combined_context += "\nRelevant error-related context:\n"
                combined_context += error_results
        
        return combined_context
        
    def _build_error_query(self, user_message: str) -> Dict[str, Any]:
        """
        Build a specialized query for searching error embeddings.
        Only includes error-relevant information.
        """
        error_query = {
            # Core error information
            'error_type': None,
            'severity': None,
            'experiment_type': None,
            'error_message': None,
            
            # Context from current errors
            'active_errors': [
                {
                    'type': error.error_type,
                    'severity': error.severity,
                    'experiment': error.experiment,
                    'message': error.message
                }
                for error in self.current_errors
            ] if self.current_errors else [],
            
            # Error intent detection
            'is_error_related': any(term in user_message.lower() 
                                  for term in ['error', 'fail', 'issue', 'wrong', 'problem']),
            
            # Current experiment context
            'current_experiment': self._get_current_experiment_type(),
            'experiment_phase': self._get_current_phase()
        }
        
        return error_query
    
    def _build_general_query(self, user_message: str) -> Dict[str, Any]:
        """
        Build a general query for searching conversation and metadata embeddings.
        Focuses on experiment flow and context.
        """
        general_query = {
            # Main query
            'user_message': user_message,
            
            # Recent conversation context (last 3 turns)
            'recent_context': self.conversation_history[-3:] if self.conversation_history else [],
            
            # Experiment context
            'experiment_type': self._get_current_experiment_type(),
            'analyzed_plots': self._track_analyzed_plots(),
            
            # Command context
            'recent_commands': [
                turn['content'] for turn in reversed(self.conversation_history)
                if 'RUN:' in turn.get('content', '')
            ][:3]  # Last 3 commands
        }
        
        return general_query
    
    def _get_current_experiment_type(self) -> Optional[str]:
        """Get the current experiment type from conversation history"""
        for turn in reversed(self.conversation_history):
            content = turn.get('content', '').lower()
            if any(exp in content for exp in ['esr', 'find_nv', 'galvo_scan', 'optimize']):
                for exp in ['esr', 'find_nv', 'galvo_scan', 'optimize']:
                    if exp in content:
                        return exp
        return None
    
    def _get_current_phase(self) -> Optional[str]:
        """Get the current experiment phase from conversation"""
        for turn in reversed(self.conversation_history[-5:]):  # Last 5 turns
            content = turn.get('content', '').lower()
            if any(phase in content for phase in ['setup', 'running', 'analysis', 'complete']):
                for phase in ['setup', 'running', 'analysis', 'complete']:
                    if phase in content:
                        return phase
        return None
    
    def _search_error_embeddings(self, error_query: Dict[str, Any]) -> str:
        """Search error embeddings with fallback strategies"""
        error_dir = os.path.join(self.embeddings_dir, 'errors')
        if not os.path.exists(error_dir):
            return ""
            
        results = []
        # Load and search error embeddings
        error_files = [f for f in os.listdir(error_dir) if f.endswith('.json')]
        
        for ef in error_files:
            with open(os.path.join(error_dir, ef)) as f:
                error_data = json.load(f)
                # Implement similarity search here
                # For now, just do basic matching
                if error_query['active_errors']:
                    for active_error in error_query['active_errors']:
                        if active_error['type'] == error_data.get('error_type'):
                            results.append(error_data['context'])
                elif error_query['is_error_related']:
                    if error_query['current_experiment'] == error_data.get('experiment'):
                        results.append(error_data['context'])
        
        return "\n".join(results) if results else ""
    
    def _search_general_embeddings(self, query, top_k=3):
        """Search through general conversation embeddings for relevant context.
        
        First tries exact matches by experiment type, then falls back to similarity search.
        """
        results = []
        
        # First try exact matches by experiment type
        general_dir = os.path.join(self.embeddings_dir, 'general')
        if os.path.exists(general_dir):
            general_files = [f for f in os.listdir(general_dir) if f.endswith('.json')]
            
            for gf in general_files:
                try:
                    with open(os.path.join(general_dir, gf)) as f:
                        general_data = json.load(f)
                        if general_data.get('metadata', {}).get('experiment_type') == self._get_current_experiment_type():
                            if 'conversation' in general_data:
                                # Extract conversation text
                                conv_text = '\n'.join([
                                    f"{msg['role']}: {msg['content']}" 
                                    for msg in general_data['conversation']
                                ])
                                results.append(conv_text)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"[RAG] Error loading general memory {gf}: {str(e)}")
        
        # If we don't have enough results, try similarity search
        if len(results) < top_k:
            embeddings_dir = os.path.join(self.embeddings_dir, 'embeddings')
            if os.path.exists(embeddings_dir):
                embeddings_files = [os.path.join(embeddings_dir, f) 
                                  for f in os.listdir(embeddings_dir) 
                                  if f.endswith('.json')]
                
                print(f"[RAG] Found {len(embeddings_files)} embedding files to search")
                self._log("rag", f"Searching {len(embeddings_files)} embedding files for query: {query}")
                
                for embedding_file in embeddings_files:
                    try:
                        print(f"[RAG] Searching file: {os.path.basename(embedding_file)}")
                        similar_contexts = search_similar(query, embedding_file, top_k=top_k)
                        if similar_contexts:
                            print(f"[RAG] Found {len(similar_contexts)} relevant contexts in {os.path.basename(embedding_file)}")
                            results.extend([ctx['text'] for ctx in similar_contexts if 'text' in ctx])
                    except Exception as e:
                        print(f"[RAG] Error searching embeddings file {embedding_file}: {str(e)}")
                        self._log("rag", f"Error searching embeddings file {embedding_file}: {str(e)}")
        
        # Return top_k results
        return "\n".join(results[:top_k]) if results else ""
    
    def save_conversation_embeddings(self):
        """
        Embed the entire conversation history and save to the embeddings directory.
        Also include references to any plots that were analyzed.
        """
        if not self.conversation_history:
            print("[Embeddings] No conversation history to save")
            return
        
        print(f"[Embeddings] Preparing to save conversation with {len(self.conversation_history)} turns")
        self._log("embeddings", f"Saving conversation with {len(self.conversation_history)} turns")
        
        # Format conversation for embedding
        conversation_text = []
        
        # Track which plots have been analyzed in this conversation
        for turn in self.conversation_history:
            role = turn["role"]
            content = turn["content"]
            conversation_data.append({
                "role": role,
                "content": content,
                "timestamp": self._current_timestamp(),
                "session_id": session_id
            })
        
        conversation_embedding = {
            "type": "conversation",
            "session_id": session_id,
            "timestamp": self._current_timestamp(),
            "turns": conversation_data,
            "summary": "\n".join([f"{turn['role']}: {turn['content']}" for turn in conversation_data])
        }
        
        # 2. Metadata Embeddings - Focus on experiment context
        available_plots = self._get_available_plots()
        analyzed_plots = self._track_analyzed_plots()
        
        metadata_embedding = {
            "type": "metadata",
            "session_id": session_id,
            "timestamp": self._current_timestamp(),
            "experiment_context": {
                "plots": [{
                    "path": plot_path,
                    "filename": os.path.basename(plot_path),
                    "type": next((ptype for ptype in ['ESR', 'FindNV', 'GalvoScan', 'Optimization'] 
                                if ptype in os.path.basename(plot_path)), 'Unknown'),
                    "analyzed": os.path.basename(plot_path) in analyzed_plots,
                    "timestamp": self._current_timestamp()
                } for plot_path in available_plots],
                "experiment_phase": self._get_current_phase(),
                "parameters": {}
            },
            "summary": f"Session with {len(available_plots)} plots, " + \
                      f"{len([p for p in available_plots if os.path.basename(p) in analyzed_plots])} analyzed"
        }
        
        # 3. Error Embeddings - Focus on error patterns and debugging
        error_embedding = {
            "type": "error",
            "session_id": session_id,
            "timestamp": self._current_timestamp(),
            "errors": [{
                "error_id": error.error_id,
                "timestamp": error.timestamp.isoformat(),
                "error_type": error.error_type,
                "severity": error.severity,
                "experiment": error.experiment,
                "message": error.message,
                "context": error.context,
                "stderr": error.stderr if error.stderr else None,
                "stdout": error.stdout if error.stdout else None,
                "related_plots": error.context.get('plot_refs', []) if error.context else []
            } for error in self.current_errors],
            "error_summary": self._generate_error_summary() if self.current_errors else "No errors"
        }
        
        # Save each embedding type separately
        try:
            base_path = os.path.join(self.embeddings_dir, timestamp)
            os.makedirs(base_path, exist_ok=True)
            
            # Save conversation embeddings
            conversation_file = os.path.join(base_path, "conversation.json")
            save_embeddings(conversation_embedding, conversation_file)
            
            # Save metadata embeddings
            metadata_file = os.path.join(base_path, "metadata.json")
            save_embeddings(metadata_embedding, metadata_file)
            
            # Save error embeddings
            error_file = os.path.join(base_path, "errors.json")
            save_embeddings(error_embedding, error_file)
            
            print(f"[Embeddings] Successfully saved all embeddings to {base_path}")
            self._log("embeddings", f"Saved embeddings for session {session_id} to {base_path}")
            
        except Exception as e:
            print(f"[Embeddings] Error saving embeddings: {str(e)}")
            self._log("embeddings", f"Error saving embeddings: {str(e)}")
    
    def _generate_error_summary(self):
        """Generate a summary of errors for the current session."""
        if not self.current_errors:
            return "No errors in this session"
            
        error_types = {}
        for error in self.current_errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            
        summary_parts = [
            f"Total errors: {len(self.current_errors)}",
            "Error types:"
        ]
        
        for error_type, count in error_types.items():
            summary_parts.append(f"- {error_type}: {count}")
            
        return "\n".join(summary_parts)


if __name__ == "__main__":
    agent = NVExperimentAgent()
    print("=== NV Experiment Agent CLI ===")
    print("Type 'exit' to quit.\n")
    
    # Print information about the embeddings directory
    print(f"[Embeddings] Using embeddings directory: {agent.embeddings_dir}")
    if os.path.exists(agent.embeddings_dir):
        existing_files = [f for f in os.listdir(agent.embeddings_dir) if f.endswith('.json')]
        print(f"[Embeddings] Found {len(existing_files)} existing embedding files")
    else:
        print(f"[Embeddings] Creating new embeddings directory")
        os.makedirs(agent.embeddings_dir, exist_ok=True)
    
    try:
        while True:
            user_in = input("You: ")
            if user_in.lower() in ["quit", "exit"]:
                print("\n[Embeddings] Saving conversation embeddings and plot metadata before exit...")
                agent.save_conversation_embeddings()
                print("Goodbye!")
                break
            agent.handle_user_input(user_in)
    except KeyboardInterrupt:
        print("\n\n[Embeddings] Detected keyboard interrupt. Saving conversation embeddings before exit...")
        agent.save_conversation_embeddings()
        print("Goodbye!")
    except Exception as e:
        print(f"\n\n[Error] An unexpected error occurred: {str(e)}")
        print("[Embeddings] Attempting to save conversation embeddings before exit...")
        try:
            agent.save_conversation_embeddings()
        except Exception as save_error:
            print(f"[Embeddings] Failed to save embeddings: {str(save_error)}")
        print("Goodbye!")

