# Dual-Mode Agent Refactoring

## Overview

The NVExperimentAgent has been refactored to support two distinct operation modes:

1. **Assistant Mode** - The agent serves as a helpful assistant to the human researcher
2. **Auto Mode** - The agent operates autonomously with minimal human intervention

## Key Changes

### 1. Constructor Modification

```python
def __init__(self, mode="assistant"):
    """
    Initialize the NVExperimentAgent with specified mode.
    
    Args:
        mode (str): Operation mode - either "assistant" or "auto"
            - "assistant": Asks for permission before actions (default)
            - "auto": Operates autonomously with minimal human intervention
    """
```

- Added `mode` parameter with "assistant" as default
- Added mode validation to ensure only "assistant" or "auto" are accepted
- Stored mode as instance variable for later use

### 2. Mode-Specific System Instructions

Created separate system instruction methods:

- `_get_assistant_mode_instruction()` - For assistant mode behavior
- `_get_auto_mode_instruction()` - For autonomous mode behavior
- `_get_system_instruction()` - Router method that selects appropriate instruction

**Key Differences in System Instructions:**

| Aspect | Assistant Mode | Auto Mode |
|--------|---------------|-----------|
| Permission | Must ask for permission before actions | Proceeds autonomously without permission |
| Human Interaction | Frequent guidance and permission requests | Minimal interaction, only when truly needed |
| Decision Making | Collaborative with human oversight | Independent with periodic reporting |
| Error Handling | Ask human for help with most issues | Only ask for help with unresolvable issues |

### 3. Permission Logic Modification

Updated `ask_human_for_permission()` method:

```python
def ask_human_for_permission(self, description: str) -> bool:
    if self.mode == "auto":
        # Automatically grant permission and log the action
        print(f"[AUTO MODE] Proceeding autonomously with: {description}")
        return True
    else:
        # Ask for permission as before
        ans = input("Grant permission? (yes/no): ").strip().lower()
        return (ans == "yes")
```

### 4. CLI Mode Selection

Enhanced the main execution to include mode selection:

```python
print("Choose your agent mode:")
print("1. Assistant Mode (asks for permission before actions)")
print("2. Auto Mode (operates autonomously)")

# Interactive mode selection
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
```

## Usage Examples

### Assistant Mode
```bash
$ python agent.py
=== NV Experiment Agent CLI ===
Choose your agent mode:
1. Assistant Mode (asks for permission before actions)
2. Auto Mode (operates autonomously)
Enter choice (1 or 2): 1

=== Agent initialized in ASSISTANT MODE ===
Assistant mode: The agent will ask for permission before taking actions.
```

### Auto Mode
```bash
$ python agent.py
=== NV Experiment Agent CLI ===
Choose your agent mode:
1. Assistant Mode (asks for permission before actions)
2. Auto Mode (operates autonomously)
Enter choice (1 or 2): 2

=== Agent initialized in AUTO MODE ===
Auto mode: The agent will operate autonomously with minimal human intervention.
It will only ask for help when encountering issues or needing clarification.
```

## Behavioral Differences

### Assistant Mode Behavior
- ✅ Asks permission before writing configuration files
- ✅ Asks permission before running experiments
- ✅ Asks permission before analyzing plots
- ✅ Provides detailed explanations and suggestions
- ✅ Waits for human approval at each step

### Auto Mode Behavior
- 🤖 Automatically writes configuration files as needed
- 🤖 Automatically runs experiments based on analysis
- 🤖 Automatically analyzes generated plots
- 🤖 Reports progress and findings periodically
- 🤖 Only asks for help when encountering unresolvable issues

## Benefits

1. **Flexibility**: Users can choose the level of automation that suits their needs
2. **Efficiency**: Auto mode enables hands-off operation for routine experiments
3. **Learning**: Assistant mode provides educational value with detailed explanations
4. **Safety**: Assistant mode ensures human oversight for critical decisions
5. **Scalability**: Auto mode allows running multiple experiments with minimal supervision

## Testing

Run the simple test to verify functionality:

```bash
python test_dual_mode_simple.py
```

This test verifies:
- Mode validation logic
- Permission handling differences
- Core functionality without requiring API keys

## Future Enhancements

Potential improvements could include:

1. **Hybrid Mode**: Combine aspects of both modes
2. **Configurable Autonomy**: Fine-tune which actions require permission
3. **Logging Differences**: Different logging verbosity for each mode
4. **Mode Switching**: Allow switching modes during runtime
5. **Safety Checks**: Additional safeguards for auto mode 