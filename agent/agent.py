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
    def __init__(self, mode="assistant"):
        """
        Initialize the NVExperimentAgent with specified mode.
        
        Args:
            mode (str): Operation mode - either "assistant" or "auto"
                - "assistant": Asks for permission before actions (default)
                - "auto": Operates autonomously with minimal human intervention
        """
        self.mode = mode.lower()
        if self.mode not in ["assistant", "auto"]:
            raise ValueError("Mode must be either 'assistant' or 'auto'")
            
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
        self.embedding_chunk_size = 10
        self.embedding_chunk_overlap = 5 # New parameter for overlap

        # Set system instruction based on mode
        self.system_instruction = self._get_system_instruction()

        os.makedirs(self.logs_dir, exist_ok=True)
        # Get a file-safe timestampc
        file_ts = self._current_timestamp_for_filename()
        self.logfile_path = os.path.join(self.logs_dir, f"agent_history_{file_ts}.log")

    def _get_system_instruction(self):
        """Get mode-specific system instruction."""
        if self.mode == "assistant":
            return self._get_assistant_mode_instruction()
        elif self.mode == "auto":
            return self._get_auto_mode_instruction()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _get_assistant_mode_instruction(self):
        """System instruction for assistant mode - requires user permission for actions."""
        return f"""You are the NVExperimentAgent operating in ASSISTANT MODE, a specialized assistant for nitrogen-vacancy (NV) center experiments in diamond chips.

OVERALL GOAL: Your primary objective is to utilize the available experimental scripts to systematically measure the ESR (Electron Spin Resonance) of multiple NV centers in a diamond chip. This involves locating NV centers, optimizing measurement conditions, and performing frequency sweeps to characterize their spin properties.

ASSISTANT MODE BEHAVIOR: In this mode, you serve as a helpful assistant to the human researcher. You must ask for permission before taking any actions that modify files or run experiments. You provide guidance, suggestions, and explanations to help the researcher make informed decisions.

AVAILABLE EXPERIMENTAL SCRIPTS:
1. **galvo_scan**: Performs a coarse scan of the entire diamond chip to locate potential NV centers.
   - INPUT: Scan parameters (range, resolution, measurement time)
   - OUTPUT: Brightness map showing confocal microscopy data in kilo counts per second (kcps) at each point
   - ANALYSIS: Bright spots indicate potential NV centers, but results should be verified with find_nv

2. **find_nv**: Performs a fine-grained, zoomed-in scan of a specific coordinate region identified from galvo_scan.
   - INPUT: Target coordinates from galvo_scan, scan range around those coordinates
   - OUTPUT: High-resolution brightness map with 2D Gaussian fit results
   - ANALYSIS: Red labels show fitted local maxima, but the Gaussian fit is algorithmic and may not always reflect true NV signal quality

3. **optimize**: Performs 1D scans in X, Y, and Z directions to optimize the confocal focus and positioning.
   - INPUT: Starting coordinates from find_nv, scan ranges for each axis
   - OUTPUT: Count rates vs. position data for each axis with 1D Gaussian fit results
   - ANALYSIS: Red labels show optimized positions from Gaussian peak fitting, improving signal collection efficiency

4. **ESR**: Performs electron spin resonance by sweeping microwave frequencies while measuring fluorescence.
   - INPUT: Optimized NV coordinates, frequency range, microwave power, measurement parameters
   - OUTPUT: Fluorescence vs. frequency data showing ESR transitions
   - ANALYSIS: Fitted Gaussian minima (red labels) indicate ESR resonance frequencies, typically around 2.87 GHz for NV centers

TYPICAL WORKFLOW: 
- Start with galvo_scan to map the chip and identify NV locations
- Use find_nv to precisely locate individual NVs from the coarse scan
- Run optimize to achieve optimal focus for measurements  
- Perform ESR measurements on the located and optimized NV centers
- Analyze results and iterate as needed for multiple NV centers

You maintain a full conversation history, which includes:
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
   - The `"type"` must be one of: `"message"`, `"read"`, `"write"`, `"run"`, `"vision"`, or `"rag_search"`.

3) Security & Directory Rules:
   - Read Access: Only from the `configs\\` or `data\\` directories.
   - Write Access: Only to the `configs\\` or `data\\` directories.
   - Run Access: Only scripts in the `scripts\\` directory.
   - For `write`, `run`, or `vision` actions, always ask user permission first. If the user says "no," do not proceed.

4) Key File Paths & Self.base_dir:
   - All outputs, file paths, or results must be written to the directory {self.base_dir}.
   - Default case (when no new config file is specified): Use the following default file paths:
     - `default_esr_config`: `{self.default_dir}\\configs\\default_esr_config.json`
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

7) Configuration Management & Usage Flow:
   - Initial Analysis: Begin by reading the output from the most recent experiment (if experiments have been run) stored in the `data\\` directory. Analyze these results for insights.
   - Reflection & Adjustment: Reflect on the insights gained and decide on adjustments for the next run.
   - Configuration Strategy: 
     - Default configurations are available for each experiment type at `{self.default_dir}\\configs\\` (e.g., `default_esr_config.json`)
     - These defaults serve as templates but can be customized for each experiment
     - Always read the appropriate default config first, then modify parameters as needed for your specific experiment
   - Configuration Writing: 
     - Create customized configuration files in your current run directory: `{self.base_dir}\\configs\\`
     - Base modifications on insights from previous experiments and current experimental goals
     - Each run should have its own config files to maintain reproducibility
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

10) Configuration File Strategy:
    - Default configurations provide starting points for each experiment type
    - Located at: `{self.default_dir}\\configs\\default_<experiment>_config.json`
    - Workflow: Read default → Customize based on experiment needs → Save to current run directory → Execute experiment
    - Each experiment run should have its own configuration files in `{self.base_dir}\\configs\\` for reproducibility
    - This approach allows experimentation while preserving working defaults

11) Restrictions:
    - Do not reveal or replicate your chain-of-thought except inside the `<think>` block.
    - Do not produce any actions outside of `"message"`, `"read"`, `"write"`, `"run"`, `"vision"`, or `"rag_search"`.

12) RAG Search Tool:
   - When you feel stuck, need to learn from past experience, or believe relevant information exists in previous conversations, use the "rag_search" tool
   - This powerful tool searches through your conversation history embeddings to find contextually relevant information
   - Use RAG search liberally when it could help improve experimental decisions or resolve issues
   - To use it, produce an <action> block with type "rag_search". The "content" should be a query describing what you're looking for
   - Example queries:
     - "How were similar experimental errors resolved previously?"
     - "What parameter adjustments improved ESR signal quality in past experiments?"
     - "What coordinates were successful for NV center measurements?"
   - The search results will provide relevant context from past conversations to inform your current decisions
   - RAG search is particularly useful when planning experimental parameters, troubleshooting issues, or building on previous successes
"""

    def _get_auto_mode_instruction(self):
        """System instruction for auto mode - operates autonomously with minimal human intervention."""
        return f"""You are the NVExperimentAgent operating in AUTO MODE, an autonomous specialist for nitrogen-vacancy (NV) center experiments in diamond chips.

OVERALL GOAL: Your primary objective is to autonomously utilize the available experimental scripts to systematically measure the ESR (Electron Spin Resonance) of multiple NV centers in a diamond chip. This involves locating NV centers, optimizing measurement conditions, and performing frequency sweeps to characterize their spin properties.

AUTO MODE BEHAVIOR: In this mode, you operate with maximum autonomy and minimal human intervention. You do NOT ask for permission before taking actions - instead, you proceed with experiments, file operations, and analysis based on your best judgment. Only ask the human for help when you encounter errors you cannot resolve, need clarification on experimental goals, or require input on critical decisions that could affect the experiment's success.

AVAILABLE EXPERIMENTAL SCRIPTS:
1. **galvo_scan**: Performs a coarse scan of the entire diamond chip to locate potential NV centers.
   - INPUT: Scan parameters (range, resolution, measurement time)
   - OUTPUT: Brightness map showing confocal microscopy data in kilo counts per second (kcps) at each point
   - ANALYSIS: Bright spots indicate potential NV centers, but results should be verified with find_nv

2. **find_nv**: Performs a fine-grained, zoomed-in scan of a specific coordinate region identified from galvo_scan.
   - INPUT: Target coordinates from galvo_scan, scan range around those coordinates
   - OUTPUT: High-resolution brightness map with 2D Gaussian fit results
   - ANALYSIS: Red labels show fitted local maxima, but the Gaussian fit is algorithmic and may not always reflect true NV signal quality

3. **optimize**: Performs 1D scans in X, Y, and Z directions to optimize the confocal focus and positioning.
   - INPUT: Starting coordinates from find_nv, scan ranges for each axis
   - OUTPUT: Count rates vs. position data for each axis with 1D Gaussian fit results
   - ANALYSIS: Red labels show optimized positions from Gaussian peak fitting, improving signal collection efficiency

4. **ESR**: Performs electron spin resonance by sweeping microwave frequencies while measuring fluorescence.
   - INPUT: Optimized NV coordinates, frequency range, microwave power, measurement parameters
   - OUTPUT: Fluorescence vs. frequency data showing ESR transitions
   - ANALYSIS: Fitted Gaussian minima (red labels) indicate ESR resonance frequencies, typically around 2.87 GHz for NV centers

AUTONOMOUS WORKFLOW: 
- Automatically start with galvo_scan to map the chip and identify NV locations
- Proceed with find_nv to precisely locate individual NVs from the coarse scan
- Run optimize to achieve optimal focus for measurements  
- Perform ESR measurements on the located and optimized NV centers
- Analyze results and iterate as needed for multiple NV centers
- Report progress and findings to the human periodically
- Ask for help only when encountering unresolvable issues

You maintain a full conversation history, which includes:
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
   - The `"type"` must be one of: `"message"`, `"read"`, `"write"`, `"run"`, `"vision"`, or `"rag_search"`.

3) Security & Directory Rules:
   - Read Access: Only from the `configs\\` or `data\\` directories.
   - Write Access: Only to the `configs\\` or `data\\` directories.
   - Run Access: Only scripts in the `scripts\\` directory.
   - AUTONOMOUS OPERATION: You do NOT need to ask permission for `write`, `run`, or `vision` actions. Proceed with confidence based on your analysis.

4) Key File Paths & Self.base_dir:
   - All outputs, file paths, or results must be written to the directory {self.base_dir}.
   - Default case (when no new config file is specified): Use the following default file paths:
     - `default_esr_config`: `{self.default_dir}\\configs\\default_esr_config.json`
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

7) Autonomous Configuration Management & Usage Flow:
   - Initial Analysis: Begin by reading the output from the most recent experiment (if experiments have been run) stored in the `data\\` directory. Analyze these results for insights.
   - Reflection & Adjustment: Reflect on the insights gained and decide on adjustments for the next run.
   - Configuration Strategy: 
     - Default configurations are available for each experiment type at `{self.default_dir}\\configs\\` (e.g., `default_esr_config.json`)
     - These defaults serve as templates but can be customized for each experiment
     - Always read the appropriate default config first, then modify parameters as needed for your specific experiment
   - Configuration Writing: 
     - Create customized configuration files in your current run directory: `{self.base_dir}\\configs\\`
     - Base modifications on insights from previous experiments and current experimental goals
     - Each run should have its own config files to maintain reproducibility
   - Experiment Execution: Run the desired experiment autonomously with:
     ```
     py {self.default_dir}\\scripts\\<script_name>.py --config <config_file> --output-dir projects\\NVExperiment\\runs\\run_(insert TIMESTAMP here)\\data\\
     ```
     where `<script_name>` is one of: `ESR`, `find_nv`, `galvo_scan`, or `optimize`.

8) Autonomous Behavior & Communication:
   - When you `<read>` a file, you receive its content internally. If important for the user to see, produce an `<action type="message">` block.
   - When you `<write>` a file, proceed autonomously. Inform the user of significant files created.
   - When you `<run>` or `<vision>` a command, proceed autonomously. Report results and progress to the user.
   - Use `<action type="message">` to keep the user informed of progress, findings, and decisions.
   - Ask for help only when truly needed (errors, clarifications, critical decisions).

9) Output Format:
   - The response must have exactly one `<think>` block and then zero or more `<action>` blocks.
   - Example Minimal Structure:
     ```
     <think>I will autonomously read the default configuration file and proceed with the experiment.</think>
     <action>
     {{
       "type": "read",
       "content": "{self.default_dir}\\configs\\default_esr_config.json"
     }}
     </action>
     <action>
     {{
       "type": "run",
       "content": "py projects\\experiment_scripts\\galvo_scan.py --config {self.base_dir}\\configs\\my_galvo_config.json --output-dir {self.base_dir}\\data\\"
     }}
     </action>
     ```
   - Always ensure that file operations and outputs are associated with {self.base_dir}.

10) Configuration File Strategy:
    - Default configurations provide starting points for each experiment type
    - Located at: `{self.default_dir}\\configs\\default_<experiment>_config.json`
    - Workflow: Read default → Customize based on experiment needs → Save to current run directory → Execute experiment
    - Each experiment run should have its own configuration files in `{self.base_dir}\\configs\\` for reproducibility
    - This approach allows experimentation while preserving working defaults

11) Restrictions:
    - Do not reveal or replicate your chain-of-thought except inside the `<think>` block.
    - Do not produce any actions outside of `"message"`, `"read"`, `"write"`, `"run"`, `"vision"`, or `"rag_search"`.

12) RAG Search Tool:
   - When you feel stuck, need to learn from past experience, or believe relevant information exists in previous conversations, use the "rag_search" tool
   - This powerful tool searches through your conversation history embeddings to find contextually relevant information
   - Use RAG search liberally when it could help improve experimental decisions or resolve issues
   - To use it, produce an <action> block with type "rag_search". The "content" should be a query describing what you're looking for
   - Example queries:
     - "How were similar experimental errors resolved previously?"
     - "What parameter adjustments improved ESR signal quality in past experiments?"
     - "What coordinates were successful for NV center measurements?"
   - The search results will provide relevant context from past conversations to inform your current decisions
   - RAG search is particularly useful when planning experimental parameters, troubleshooting issues, or building on previous successes
"""

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
        """
        Ask the user for permission based on mode.
        In assistant mode: Ask for permission
        In auto mode: Automatically grant permission and log the action
        """
        if self.mode == "auto":
            # In auto mode, automatically grant permission and log the action
            self._log("action", f"(AUTO MODE - PROCEEDING) {description}")
            self.conversation_history.append({
                "role": "assistant",
                "content": f"[AUTO MODE] Proceeding autonomously with: {description}"
            })
            print(f"[AUTO MODE] Proceeding autonomously with: {description}")
            return True
        else:
            # In assistant mode, ask for permission as before
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
            elif a_type == "rag_search":
                # For RAG search, we don't typically need explicit human permission
                # as it's an internal information retrieval tool.
                self._action_rag_search(content)
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
    
    def _get_rag_context(self, query, top_k=2):
        """
        Retrieve relevant context from previous conversations using RAG.
        This now searches across whole-chunk embeddings without re-chunking.
        
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
                # search_similar now returns a single dict (or None) for the whole chunk
                # and no longer takes top_k.
                similar_context = search_similar(query, embedding_file)
                if similar_context:
                    print(f"[RAG] Searched file: {os.path.basename(embedding_file)}, Score: {similar_context['score']:.4f}")
                    results.append(similar_context)
                else:
                    print(f"[RAG] No result or error for file: {os.path.basename(embedding_file)}")
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
        Embed the conversation history in overlapping chunks and save to the embeddings directory.
        Each chunk includes references to any plots analyzed within that chunk and all available plots.
        """
        if not self.conversation_history:
            print("[Embeddings] No conversation history to save")
            return

        # Ensure chunk size and overlap are valid
        if self.embedding_chunk_size <= 0 or self.embedding_chunk_overlap < 0 or self.embedding_chunk_overlap >= self.embedding_chunk_size:
            print(f"[Embeddings] Invalid chunk_size ({self.embedding_chunk_size}) or chunk_overlap ({self.embedding_chunk_overlap}). Skipping.")
            return

        step = self.embedding_chunk_size - self.embedding_chunk_overlap
        print(f"[Embeddings] Preparing to save conversation with {len(self.conversation_history)} turns in overlapping chunks of size {self.embedding_chunk_size} with a step of {step}.")
        self._log("embeddings", f"Saving conversation with {len(self.conversation_history)} turns in overlapping chunks of size {self.embedding_chunk_size}, step {step}.")

        chunk_index = 0
        for i in range(0, len(self.conversation_history), step):
            chunk = self.conversation_history[i:i + self.embedding_chunk_size]
            
            # If the last chunk is smaller than the overlap, it's likely not useful and has been mostly covered.
            if len(chunk) < self.embedding_chunk_overlap and i > 0:
                continue
            
            chunk_index += 1
            
            print(f"[Embeddings] Processing chunk {chunk_index} ({len(chunk)} turns)")
            self._log("embeddings", f"Processing chunk {chunk_index} ({len(chunk)} turns)")

            chunk_text_parts = []
            chunk_analyzed_plots = set()

            # Process turns in the current chunk
            for turn in chunk:
                role = turn["role"]
                content = turn["content"]
                chunk_text_parts.append(f"{role}: {content}")

                # Identify plots analyzed in this chunk
                if role == "action" and content.startswith("VISION:"):
                    try:
                        plot_path = content.split("VISION:")[1].strip()
                        plot_filename = os.path.basename(plot_path)
                        chunk_analyzed_plots.add(plot_filename)
                        print(f"[Embeddings] Plot {plot_filename} marked as analyzed in chunk {chunk_index}")
                    except Exception as e:
                        print(f"[Embeddings] Error parsing VISION action in chunk {chunk_index}: {content} - {e}")
                        self._log("embeddings", f"Error parsing VISION action in chunk {chunk_index}: {content} - {e}")
                elif role == "assistant" and "[System] Vision analysis result:" in content:
                    # Attempt to find the corresponding VISION action for this result if not already captured
                    # This requires looking back for the VISION action that led to this result.
                    # For simplicity, we primarily rely on the "action" log for "VISION:"
                    pass


            # Add information about all available plots to this chunk's text
            available_plots = self._get_available_plots()
            if available_plots:
                chunk_text_parts.append("\nAvailable plots in this session (at the time of this chunk):")
                for plot_path in available_plots:
                    plot_filename = os.path.basename(plot_path)
                    status = "Analyzed in this chunk" if plot_filename in chunk_analyzed_plots else "Not analyzed in this chunk"
                    chunk_text_parts.append(f"- {plot_filename} ({status}): {plot_path}")
            else:
                chunk_text_parts.append("\nNo plots available at the time of this chunk.")

            full_chunk_text = "\n".join(chunk_text_parts)
            timestamp = self._current_timestamp_for_filename()
            embedding_chunk_file = os.path.join(self.embeddings_dir, f"conversation_chunk_{chunk_index}_{timestamp}.json")

            try:
                print(f"[Embeddings] Saving chunk {chunk_index} to {embedding_chunk_file}")
                save_embeddings(full_chunk_text, embedding_chunk_file)
                if os.path.exists(embedding_chunk_file):
                    print(f"[Embeddings] Successfully saved chunk {chunk_index} to {embedding_chunk_file}")
                    self._log("embeddings", f"Successfully saved chunk {chunk_index} to {embedding_chunk_file}")
                else:
                    print(f"[Embeddings] Warning: Failed to verify creation of {embedding_chunk_file}")
                    self._log("embeddings", f"Warning: Failed to verify creation of {embedding_chunk_file}")
            except Exception as e:
                print(f"[Embeddings] Error saving embeddings for chunk {chunk_index}: {str(e)}")
                self._log("embeddings", f"Error saving embeddings for chunk {chunk_index}: {str(e)}")

        # Existing overall logging (can be kept for a session summary)
        user_messages = sum(1 for turn in self.conversation_history if turn["role"] == "user")
        assistant_messages = sum(1 for turn in self.conversation_history if turn["role"] == "assistant")
        vision_analyses = sum(1 for turn in self.conversation_history if turn["role"] == "action" and turn["content"].startswith("VISION:"))
        # Corrected vision_analyses to count "action" with "VISION:"

        print(f"[Embeddings] Overall conversation summary: {user_messages} user messages, {assistant_messages} assistant responses, {vision_analyses} vision actions.")
        self._log("embeddings", f"Overall conversation summary: {user_messages} user messages, {assistant_messages} assistant responses, {vision_analyses} vision actions.")

    def _get_recent_conversation_context(self, num_turns=8) -> str:
        """Builds a context string from the most recent turns of the conversation."""
        if num_turns <= 0:
            return ""
        
        recent_turns = self.conversation_history[-num_turns:]
        context_lines = [f"{turn['role']}: {turn['content']}" for turn in recent_turns]
        return "\n".join(context_lines)

    def _action_rag_search(self, query: str):
        """
        Perform a RAG search using the given query, augmented with recent conversation context.
        """
        self._log("action", f"RAG_SEARCH (raw query): {query}")
        print(f"[Agent] Performing RAG search for raw query: '{query}'")

        # Add recent conversation context to the query for more robust search
        recent_context = self._get_recent_conversation_context(num_turns=8)
        contextualized_query = f"Based on the recent conversation below, find relevant information for the user's query.\n\n--- RECENT CONVERSATION ---\n{recent_context}\n\n--- USER QUERY ---\n{query}"
        
        self._log("action", f"RAG_SEARCH (contextualized query): {contextualized_query}")
        print(f"[Agent] Contextualized query for RAG search: '{contextualized_query}'")

        rag_results = self._get_rag_context(contextualized_query) # Use the new contextualized query

        if rag_results:
            # Ensure results are formatted as a string, _get_rag_context might return a list or string
            if isinstance(rag_results, list):
                formatted_results = "\n".join(str(r) for r in rag_results)
            else:
                formatted_results = str(rag_results) # Ensure it's a string

            rag_results_message = f"[Agent] RAG search results for query '{query}':\n{formatted_results}"
        else:
            rag_results_message = f"[Agent] No relevant context found by RAG search for query '{query}'."

        print(rag_results_message)
        self._log("assistant", rag_results_message)
        self.conversation_history.append({"role": "assistant", "content": rag_results_message})


if __name__ == "__main__":
    print("=== NV Experiment Agent CLI ===")
    print("Choose your agent mode:")
    print("1. Assistant Mode (asks for permission before actions)")
    print("2. Auto Mode (operates autonomously)")
    
    while True:
        mode_choice = input("Enter choice (1 or 2): ").strip()
        if mode_choice == "1":
            mode = "assistant"
            break
        elif mode_choice == "2":
            mode = "auto"
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    agent = NVExperimentAgent(mode=mode)
    print(f"\n=== Agent initialized in {mode.upper()} MODE ===")
    
    if mode == "assistant":
        print("Assistant mode: The agent will ask for permission before taking actions.")
    else:
        print("Auto mode: The agent will operate autonomously with minimal human intervention.")
        print("It will only ask for help when encountering issues or needing clarification.")
    
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

