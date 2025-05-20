import os
import json
import re
import subprocess
from datetime import datetime

from anthropic_engine import call_llm, call_vision  # Import both text and vision functions
# from deepseek_engine import call_llm, call_vision  # Import both text and vision functions
#from anthropic_engine import call_vision
#from deepseek_engine import call_llm
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
        # Get a file-safe timestampc
        file_ts = self._current_timestamp_for_filename()
        self.logfile_path = os.path.join(self.logs_dir, f"agent_history_{file_ts}.log")

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
        self.conversation_history.append({"role": "user", "content": user_message})
        
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


    def _action_run_command(self, command: str):
        """
        Run a shell command (only if in 'scripts\\' directory and is one of the allowed scripts), capturing output.
        """
        self._log("action", f"RUN: {command}")
        try:
            parsed = self._parse_run_command(command)
            allowed_scripts = [f"{self.default_dir}\\scripts\\ESR.py", f"{self.default_dir}\\scripts\\find_nv.py", 
                               f"{self.default_dir}\\scripts\\galvo_scan.py", f"{self.default_dir}\\scripts\\optimize.py"]
            script_normalized = parsed["script"].replace("/", "\\")

            if script_normalized not in allowed_scripts:
                raise ValueError("Command not allowed: script not among allowed options.")
            
            # Ensure the data directory exists
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Log the command as is - the agent should have included the output directory
            self._log("action", f"Running command: {command}")
            
            result = subprocess.run(command, shell=True, check=True, capture_output=True)
            stdout_text = result.stdout.decode()
            stderr_text = result.stderr.decode()
            out_msg = "[System] Command output:\n" + stdout_text
            if stderr_text:
                out_msg += "\n[System] Command errors:\n" + stderr_text
            print(out_msg)
            self._log("action", f"RUN OUTPUT: {stdout_text}")
            self.conversation_history.append({"role": "assistant", "content": out_msg})
        except Exception as e:
            err_msg = f"[System] Error running command: {str(e)}"
            print(err_msg)
            self._log("action", err_msg)
            self.conversation_history.append({"role": "assistant", "content": err_msg})

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
    
    def _get_rag_context(self, query, top_k=3):
        """
        Retrieve relevant context from previous conversations using RAG.
        Highlight any plot references in the retrieved context.
        
        Args:
            query: The user's query to search against
            top_k: Number of most relevant contexts to retrieve
            
        Returns:
            String containing the most relevant contexts with plot references highlighted
        """
        # Check if embeddings directory exists and has files
        if not os.path.exists(self.embeddings_dir):
            print(f"[RAG] Embeddings directory {self.embeddings_dir} does not exist")
            return ""
            
        # Load all embeddings from the embeddings directory
        embeddings_files = [os.path.join(self.embeddings_dir, f) 
                            for f in os.listdir(self.embeddings_dir) 
                            if f.endswith('.json')]
        
        if not embeddings_files:
            print(f"[RAG] No embedding files found in {self.embeddings_dir}")
            return ""
        
        print(f"[RAG] Found {len(embeddings_files)} embedding files to search")
        self._log("rag", f"Searching {len(embeddings_files)} embedding files for query: {query}")
        
        # Search for similar contexts across all embedding files
        results = []
        for embedding_file in embeddings_files:
            try:
                print(f"[RAG] Searching file: {os.path.basename(embedding_file)}")
                similar_contexts = search_similar(query, embedding_file, top_k=top_k)
                if similar_contexts:
                    print(f"[RAG] Found {len(similar_contexts)} relevant contexts in {os.path.basename(embedding_file)}")
                    results.extend(similar_contexts)
                else:
                    print(f"[RAG] No relevant contexts found in {os.path.basename(embedding_file)}")
            except Exception as e:
                print(f"[RAG] Error searching embeddings file {embedding_file}: {str(e)}")
                self._log("rag", f"Error searching embeddings file {embedding_file}: {str(e)}")
        
        # Sort by similarity score and take top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]
        
        # Format the results
        if not results:
            print(f"[RAG] No relevant contexts found after searching all embedding files")
            return ""
        
        print(f"[RAG] Found {len(results)} relevant contexts after filtering by similarity")
        
        context_text = []
        for i, result in enumerate(results):
            # Check if the result contains plot references
            text = result["text"]
            plot_references = result.get("plot_references", [])
            
            # Log the result
            print(f"[RAG] Context {i+1}/{len(results)}: Similarity score {result['score']:.2f}")
            if plot_references:
                print(f"[RAG] Context {i+1} contains {len(plot_references)} plot references")
            
            # Add the context with plot references highlighted
            context_entry = f"Context (similarity: {result['score']:.2f}):\n{text}"
            if plot_references:
                context_entry += "\n\nRelevant plot references:\n" + "\n".join(plot_references)
            
            context_text.append(context_entry)
        
        return "\n\n".join(context_text)
    
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
        analyzed_plots = set()
        
        # Count message types for logging
        user_messages = 0
        assistant_messages = 0
        vision_analyses = 0
        
        for turn in self.conversation_history:
            role = turn["role"]
            content = turn["content"]
            conversation_text.append(f"{role}: {content}")
            
            # Count message types
            if role == "user":
                user_messages += 1
            elif role == "assistant":
                assistant_messages += 1
            
            # Check if this is a vision analysis result
            if role == "assistant" and "[System] Vision analysis result:" in content:
                vision_analyses += 1
                # Extract the plot filename from previous messages
                for i in range(len(self.conversation_history)):
                    if (i < len(self.conversation_history) - 1 and 
                        "VISION:" in self.conversation_history[i].get("content", "")):
                        plot_path = self.conversation_history[i]["content"].split("VISION:")[1].strip()
                        plot_filename = os.path.basename(plot_path)
                        analyzed_plots.add(plot_filename)
        
        print(f"[Embeddings] Conversation summary: {user_messages} user messages, {assistant_messages} assistant responses, {vision_analyses} vision analyses")
        
        # Add references to all plots in the data directory
        available_plots = self._get_available_plots()
        if available_plots:
            print(f"[Embeddings] Including {len(available_plots)} plots in embedding data")
            conversation_text.append("\nAvailable plots in this session:")
            for plot_path in available_plots:
                plot_filename = os.path.basename(plot_path)
                status = "Analyzed" if plot_filename in analyzed_plots else "Not analyzed"
                conversation_text.append(f"- {plot_filename} ({status}): {plot_path}")
        else:
            print("[Embeddings] No plots available to include in embedding data")
        
        # Join all turns with newlines
        full_text = "\n".join(conversation_text)
        
        # Generate a timestamp for the embedding file
        timestamp = self._current_timestamp_for_filename()
        embedding_file = os.path.join(self.embeddings_dir, f"conversation_{timestamp}.json")
        
        # Save the embeddings
        try:
            # Count existing embedding files before saving
            existing_files = [f for f in os.listdir(self.embeddings_dir) if f.endswith('.json')]
            print(f"[Embeddings] Current embedding files count: {len(existing_files)}")
            
            # Save the new embeddings
            save_embeddings(full_text, embedding_file)
            
            # Verify the file was created
            if os.path.exists(embedding_file):
                print(f"[Embeddings] Successfully saved conversation to {embedding_file}")
                self._log("embeddings", f"Successfully saved conversation to {embedding_file}")
                
                # Count embedding files after saving to confirm a new one was added
                updated_files = [f for f in os.listdir(self.embeddings_dir) if f.endswith('.json')]
                print(f"[Embeddings] Updated embedding files count: {len(updated_files)}")
                if len(updated_files) > len(existing_files):
                    print("[Embeddings] Confirmed: New embedding file was created")
                else:
                    print("[Embeddings] Warning: No new embedding file was created")
            else:
                print(f"[Embeddings] Warning: Failed to verify creation of {embedding_file}")
        except Exception as e:
            print(f"[Embeddings] Error saving embeddings: {str(e)}")
            self._log("embeddings", f"Error saving embeddings: {str(e)}")


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

