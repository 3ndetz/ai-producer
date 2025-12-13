SYSTEM_READY_MESSAGE = """✓ System ready!

**How it works:**
1. **Planner Agent**: Decides whether to continue generation by evaluating issue resolvability and repetition, providing key focus details.
2. **Generator Agent**: Composes image generation prompts from user input, Planner recommendations, and Critic feedback and generate new image!
3. **Critic Agent**: Evaluates image-prompt alignment, describing the image and highlighting correct/incorrect elements.

You can provide feedback during interruptions to guide the process.

**Example requests:**
- "A cat conducting an orchestra of mice in a submarine"
- "Elephants skydiving with umbrellas over a volcano"
- "Penguins running a bakery on the moon"

Type your image request below to begin!
"""

INITIALIZING_MESSAGE = "Initializing AI Producer..."

INIT_FAILED_MESSAGE_TEMPLATE = "❌ Initialization failed: {}\n\nPlease check your .env configuration."

WELCOME_BACK_MESSAGE = "Welcome back! Describe the image you'd like to create."

AGENT_ERROR_MESSAGE_TEMPLATE = """❌ **{} Agent Error**

**Iteration**: {}

**Error Details**:
```
{}
```
**Action**: Workflow has been stopped. Please check the error and try again.
"""


