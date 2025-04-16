import os
import json
import re
import subprocess
from datetime import datetime

from anthropic_engine import call_llm, call_vision  # Import both text and vision functions
# from deepseek_engine import call_llm, call_vision  # Import both text and vision functions
#from anthropic_engine import call_vision
#from deepseek_engine import call_llm

class NVExperimentAgent:
    def __init__(self):
        self.project_root_dir = 'projects'
        self.project_name = 'NVExperiment'
        self.runs_dir_name = 'runs'
        self.run_dir_prefix = 'run'

        file_ts = self._current_timestamp_for_filename()
        self.run_dir = f"{self.run_dir_prefix}_{file_ts}"
        self.base_dir = os.path.join(self.project_root_dir, self.project_name, self.runs_dir_name, self.run_dir)

        self.default_config_dir = os.path.join(self.project_root_dir, 'configs')
        
        self.config_dir = os.path.join(self.base_dir, "configs")
        self.data_dir   = os.path.join(self.base_dir, "data")
        self.logs_dir   = os.path.join(self.base_dir, "logs")
        self.scripts_dir = "experiment_scripts"  # updated directory for scripts

        # The entire conversation history (user messages, assistant messages, actions, etc.)
        self.conversation_history = []

        # Extended system instruction with updated run command and vision option details
        self.system_instruction = (
            """You are the NVExperimentAgent. You maintain a full conversation history, which includes:

- All user messages,
- All assistant messages (your own),
- All actions you have taken (read/write/run/vision),
- The results of those actions.

You have the following constraints and abilities:

1) You must produce exactly one <think> block per response, containing your private chain-of-thought. 
   - Do not reveal that chain-of-thought to the user except within the <think>…</think> block (which the system may hide).
2) You may produce zero or more <action> blocks, each containing valid JSON.
   - The <action> block must have the form:
       <action>
       {
         "type": "...",
         "content": ...
       }
       </action>
   - The "type" must be one of: "message", "read", "write", "run", "vision".
3) Security & Directory Rules:
   - read only from configs/ or data/
   - write only to configs/ or data/
   - run only scripts in experiment_scripts/
   - For write, run, or vision actions, always ask user permission first. If the user says “no,” do not proceed.
4) Key File Paths:
   - default_esr_config: projects\\configs\\default_esr_config.json
   - default_find_nv_config: projects\\configs\\default_find_nv_config.json
   - default_galvo_scan_config: projects\\configs\\default_galvo_scan_config.json
   - default_optimize_config: projects\\configs\\default_optimize_config.json
   - ESR script: projects\\experiment_scripts\\ESR.py
   - find_nv script: projects\\experiment_scripts\\find_nv.py
   - galvo_scan script: projects\\experiment_scripts\\galvo_scan.py
   - optimize script: projects\\experiment_scripts\\optimize.py
     Typically run as:
       py projects\\experiment_scripts\\ESR.py --config <some_config_file>
       py projects\\experiment_scripts\\find_nv.py --config <some_config_file>
       py projects\\experiment_scripts\\galvo_scan.py --config <some_config_file>
       py projects\\experiment_scripts\\optimize.py --config <some_config_file>
    - galvo scan is a coarse version of looking for NVs in field, whereas findnv is a finer version 
5) Run Command Options:
   - The run command must include one of the following four options: ESR, find_nv, galvo_scan, or optimize.
   - IMPORTANT: You MUST include the --output-dir parameter in your command to specify where results should be saved.
   - Always use the current run's data directory as the output directory: projects/NVExperiment/runs/run_(insert TIMESTAMP here)/data/
   - The complete command format should be:
         py projects/experiment_scripts/<script_name>.py --config <config_file> --output-dir projects/NVExperiment/runs/run_(insert TIMESTAMP here)/data/
     where <script_name> is one of ESR, find_nv, galvo_scan, or optimize.
6) Vision Option:
   - In addition to running scripts, you can analyze plot images.
   - Use the command: vision <plot_file_path>
   - IMPORTANT: All experiment outputs are automatically saved to the current run's data directory (projects/NVExperiment/runs/run_(insert TIMESTAMP here)/data/).
   - When analyzing plots, simply use the base filename (e.g., 'vision ESR_plot.png') - the system will automatically locate it in the current run's data directory.
   - DO NOT use paths like 'projects/data/ESR_plot.png' as these are incorrect. The correct plots are always in the run-specific data directory.
   - When reading GalvoScan_plot.png, NVs are accociate with large bright dots, estimate and read out the center coordinates of bright dots for next steps.
7) Typical usage flow for ESR, can be migrated to doing any experiments by replacing 'esr' with the experiment:
   - Begin by reading the output from the most recent experiment (if you have already run experiments) stored in the current run's data directory. Carefully analyze these results to identify results and areas for improvement.
   - Reflect on the insights gained from the previous experiment and decide on the adjustments needed for the upcoming run.
   - Read the default config at projects/configs/default_esr_config.json.
   - Write a new config, e.g. projects/configs/my_new_experiment.json, changing relevant fields based on the insight from the previous experiment.
   - Run the command with the output directory explicitly specified: 
         py projects/experiment_scripts/ESR.py --config configs/my_new_experiment.json --output-dir projects/NVExperiment/runs/run_(insert TIMESTAMP here)/data/
   - After the experiment completes, analyze the results using 'vision ESR_plot.png' to see the plot that was generated.
8) Behavior:
   - If you <read> a file, you receive its content internally. If you want the user to see it, produce a <action type="message"> block.
   - If you <write> a file, ask user permission. If denied, do not write.
   - If you <run> or <vision> a command, ask user permission. If denied, do not proceed.
   - You can use <message> to communicate with the user.
9) Output Format:
   - Exactly one <think> block, then zero or more <action> blocks.
   - Example minimal structure for read, write, and run experiment:
       <think>I will read the config</think>
       <action>
       {
         "type": "read",
         "content": "projects\\configs\\default_esr_config.json"
       }
       </action>
       <action>
       {
         "type": "write",
         "content": {
           "path": <path to write the config>,
           "data": <mimic the config dict you read>
         }
       }
       </action>
   - You can add more actions as needed.
10) Do not reveal or replicate your chain-of-thought except inside <think>.
    Do not produce any actions outside "message", "read", "write", "run", or "vision".

End of system instructions.
"""
        )

        os.makedirs(self.logs_dir, exist_ok=True)
        # Get a file-safe timestamp
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
        Combine system_instruction and conversation_history into a single prompt.
        """
        prompt = self.system_instruction.strip()
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
        return (ans == "yes")

    def handle_user_input(self, user_message: str):
        """
        Process user prompt: log it, build the prompt, call the LLM, parse and execute actions.
        """
        self._log("user", user_message)
        self.conversation_history.append({"role": "user", "content": user_message})
        full_prompt = self._build_prompt()
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
        Read a file from allowed directories (configs/ or data/).
        """
        self._log("action", f"READ: {filepath}")
        allowed_prefixes = [self.config_dir, self.default_config_dir, self.data_dir]

        print(f"ALLOWED PREFIXES: {allowed_prefixes}")

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
        Write JSON data to a file in allowed directories (configs/ or data/).
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
            py experiment_scripts/<script_name>.py --config <config_file> [--output-dir <output_directory>]
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
        Run a shell command (only if in 'experiment_scripts/' directory and is one of the allowed scripts), capturing output.
        """
        self._log("action", f"RUN: {command}")
        try:
            parsed = self._parse_run_command(command)
            allowed_scripts = ["experiment_scripts/ESR.py", "experiment_scripts/find_nv.py", "experiment_scripts/galvo_scan.py", "experiment_scripts/optimize.py", 
                           "NVExperiment/scripts/ESR.py", "NVExperiment/scripts/find_nv.py", "NVExperiment/scripts/galvo_scan.py", "NVExperiment/scripts/optimize.py"]
            script_normalized = parsed["script"].replace("\\", "/")
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
        Analyze a plot image from the data/ directory using the vision model, including conversation context.
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
        

if __name__ == "__main__":
    agent = NVExperimentAgent()
    print("=== NV Experiment Agent CLI ===")
    print("Type 'exit' to quit.\n")
    while True:
        user_in = input("You: ")
        if user_in.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        agent.handle_user_input(user_in)

