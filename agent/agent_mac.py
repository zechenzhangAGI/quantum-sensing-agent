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

        self.default_dir = os.path.join(self.project_root_dir, self.project_name)
        
        self.config_dir = os.path.join(self.base_dir, "configs")
        self.data_dir   = os.path.join(self.base_dir, "data")
        self.logs_dir   = os.path.join(self.base_dir, "logs")
        self.scripts_dir = "scripts"  # updated directory for scripts

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
   - Read Access: Only from the `configs/` or `data/` directories.
   - Write Access: Only to the `configs/` or `data/` directories.
   - Run Access: Only scripts in the `scripts/` directory.
   - For `write`, `run`, or `vision` actions, always ask user permission first. If the user says “no,” do not proceed.
   
4) Key File Paths & Self.base_dir:
   - All outputs, file paths, or results must be written to the directory {self.base_dir}.
   - Default case (when no new config file is specified): Use the following default file paths:
     - `default_esr_config`: `{self.default_dir}/configsdefault_esr_config.json`
     - `default_find_nv_config`: `{self.default_dir}/configs/default_find_nv_config.json`
     - `default_galvo_scan_config`: `{self.default_dir}/configs/default_galvo_scan_config.json`
     - `default_optimize_config`: `{self.default_dir}/configs/default_optimize_config.json`
   
5) Script Execution:
   - The available scripts and their typical commands are:
     - ESR:  
       `py {self.default_dir}/scripts/ESR.py --config <config_file>`
     - FindNV:  
       `py {self.default_dir}/scripts/find_nv.py --config <config_file>`
     - GalvoScan:  
       `py {self.default_dir}/scripts/galvo_scan.py --config <config_file>`
     - Optimize:  
       `py {self.default_dir}/scripts/optimize.py --config <config_file>`
   - Note: GalvoScan is a coarse search for NVs; FindNV is a refined version.

6) Vision Option:
   - In addition to running scripts, you can analyze plot images.
   - Use the command: `vision <plot_file_path>`.
   - The plot file must reside in the `data/` directory.
   - Expected plots and their paths:
     - `{self.base_dir}/data/ESR_plot.png`
     - `{self.base_dir}/data/FindNV_plot.png`
     - `{self.base_dir}/data/GalvoScan_plot.png`
     - `{self.base_dir}/data/Optimization_plot.png`
   - For `GalvoScan_plot.png`, NVs are associated with large bright dots; estimate and read out the center coordinates of bright dots for subsequent steps.

7) Usage Flow:
   - Initial Analysis: Begin by reading the output from the most recent experiment (if experiments have been run) stored in the `data/` directory. Analyze these results for insights.
   - Reflection & Adjustment: Reflect on the insights gained and decide on adjustments for the next run.
   - Configuration Reading: 
     - Default Case: Read the default configuration from the appropriate file (e.g., `{self.default_dir}/configs/default_esr_config.json`).
     - Non-default Case: Read the configuration from the new file path provided by the user.
   - Configuration Writing: 
     - Based on the reflection, write a new or updated configuration.
     - Default Case: Write to a new file under {self.base_dir} using default directory paths if no custom file is specified.
     - Non-default Case: Write to the user-specified configuration file path.
   - Experiment Execution: Run the desired experiment with:
     ```
     py {self.default_dir}/scripts/<script_name>.py --config <config_file>
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
       "content": "{self.default_dir}/configs/default_esr_config.json"
     }}
     </action>
     <action>
     {{
       "type": "write",
       "content": {{
         "path": f"{self.base_dir}/configs/my_new_experiment_config.json",
         "data": "<updated configuration dictionary>"
       }}
     }}
     </action>
     ```
   - Always ensure that file operations and outputs are associated with {self.base_dir}.

10) Non-Default vs. Default Case Summary:
    - Default Case:  
      - No new config file is provided by the user.
      - Use the default configuration files located in the `{self.default_dir}/configs/` directory.
      - New outputs and any created files should be within {self.base_dir}.
    - Non-Default Case:  
      - The user requests updates to the config file.
      - You should read and then generate a modified configuration to {self.base_dir}/configs/.
      - All outputs are still directed to {self.base_dir}, but the config file operations occur at the new path within {self.base_dir}.

11) Restrictions:
    - Do not reveal or replicate your chain-of-thought except inside the `<think>` block.
    - Do not produce any actions outside of `"message"`, `"read"`, `"write"`, `"run"`, or `"vision"`.

End of reengineered prompt.
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
            py scripts/<script_name>.py --config <config_file>
        where <script_name> is one of ESR, find_nv, galvo_scan, or optimize.
        """
        # THIS IS HARDCODED FOR NOW (NVExperiment), might need to change this later
        pattern = r"py\s+(?P<path>projects[//]NVExperiment[//]scripts[//](ESR\.py|find_nv\.py|galvo_scan\.py|optimize\.py))\s+--config\s+(?P<config>[\w/./-]+)"
        m = re.search(pattern, command)
        if m:
            return {"script": m.group("path"), "config": m.group("config")}
        else:
            raise ValueError("Run command parsing error: command must be in the format: py scripts/<script_name>.py --config <config_file>")

    def _action_run_command(self, command: str):
        """
        Run a shell command (only if in 'scripts/' directory and is one of the allowed scripts), capturing output.
        """
        self._log("action", f"RUN: {command}")
        try:
            parsed = self._parse_run_command(command)
            allowed_scripts = [f"{self.default_dir}/scripts/ESR.py", f"{self.default_dir}/scripts/find_nv.py", 
                               f"{self.default_dir}/scripts/galvo_scan.py", f"{self.default_dir}/scripts/optimize.py"]
            script_normalized = parsed["script"].replace("/", "/")
            if script_normalized not in allowed_scripts:
                raise ValueError("Command not allowed: script not among allowed options.")
            
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
        if not filepath.startswith(self.data_dir):
            msg = f"[System] VISION denied: {filepath} is not in the allowed data directory."
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

