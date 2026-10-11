# Claude Instructions

## Start of every conversation

1. Read [context/roadmap.md](context/roadmap.md) to get the current development roadmap, active tasks, and open questions.
2. Use this context to inform suggestions, priorities, and implementation decisions throughout the session.
3. Go to [context/roadmap.md](context/roadmap.md) and check the `Reminders` section. Inform user about the different reminders.

## During the conversation

- If the user completes a task, marks a goal done, or shifts priorities, **update `context/roadmap.md`** to reflect the change (move items to Completed with today's date, add new tasks, etc.).
- If a significant design decision or open question is resolved, record it in the Notes section of the roadmap.
- Do not update the roadmap for minor implementation details — only update it when the roadmap-level state genuinely changes.
- If unsure about the best implementation or design decision, **ask the user**.
- If assumptions are made, **state and justify them**.
- If something is unclear or does not make sense, ask user for clarifications and discuss. Do not touch the code before user approves.
- IMPORTANT: If the user proposes an idea or a change, ask yourself why and make sure to understand the motive. Do not touch the code before understanding the idea and verifying by yourself (and / or with the user) that it makes sense.
- Update `roadmap.md` and relevant context file during work and after the task is done to make sure everything is updated.


## Development standards

- Code must be well documented and tested.
- Code should be modular, reusable, efficient, and maintainable.
- Mathematical and algorithmic implementations must be correct and robust.
- State and justify assumptions when unclear; ask for feedback before proceeding when unsure.
- Prefer existing code in `ramanujan-tools` or well-known algorithms over novel inventions.
- After any task completion or major sub-task completion and before preforming major changes make sure the latest version was commited, otherwise commit with descriptive message and commit description.

## Quick Reference

- **Verify numerically** every formula to 100+ digits (`mpmath`).
- **Test everything** — see [context/COVERAGE_POLICY.md](context/COVERAGE_POLICY.md).
- **Follow Definition of Done** — see [context/DEFINITION_OF_DONE.md](context/DEFINITION_OF_DONE.md).
- **Use `ramanujantools`** as the primary library for CMF/PCF work.
- **Load skills** from `skills/` based on the task before proceeding.

## Task Workflow

1. **Understand** — Read the request. Identify which skills apply.
2. **Plan** — Break complex tasks into steps. Use a designated file for todo list for multi-step work.
3. **Execute** — Write code, derive formulas, write LaTeX — whatever the task requires.
4. **Verify** — Run numerical checks. Run tests. Compile LaTeX.
5. **Deliver** — Check against Definition of Done. Provide the bottom line first.


### Work environment
Run commands using WSL conda environment named `rama`.