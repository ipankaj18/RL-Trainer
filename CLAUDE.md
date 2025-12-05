# Claude.md - RL Trainer AI Trading System

## Project Overview

**RL Trainer** is a production-ready reinforcement learning trading system for futures markets with strict Apex Trader Funding compliance. Three-phase curriculum learning: PPO + optional LLM integration.

**Version**: 1.0.0 (October 2025)
**Author**: Javier ([@javiertradess](https://x.com/javiertradess))
**License**: Apache License 2.0 (inherited from TensorTrade)
**Python**: 3.11+ (3.13 supported)

**⚠️ CRITICAL WORKFLOW NOTE**: At the start of EVERY new chat session, Claude MUST read [changelog.md](changelog.md) first (last 3 days if too large) to understand recent changes. After completing major changes, update changelog immediately.

## Core Purpose

Train AI agents to trade 8 futures markets (ES, NQ, YM, RTY, MNQ, MES, M2K, MYM) profitably with **100% Apex compliance**:
- $2,500 max trailing drawdown
- 4:59 PM ET mandatory position close
- No overnight positions
- 1+ trading days minimum for evaluation

## Technology Stack

- **RL Framework**: Stable Baselines3 (PPO, MaskablePPO)
- **Environment**: Gymnasium
- **Deep Learning**: PyTorch 2.0+, TensorFlow 2.15+
- **Trading Library**: TensorTrade (custom fork)
- **Data Processing**: Pandas, NumPy, SciPy, Scikit-Learn
- **LLM (Phase 3)**: Phi-3-mini-4k-instruct
- **Monitoring**: TensorBoard, Custom callbacks

## Intelligent Tool Selection Framework

**CRITICAL**: Claude automatically uses MCP servers, Skills, and Subagents based on task context WITHOUT explicit user invocation.

### Available Tools

#### MCP Servers
1. **context7** - Up-to-date library documentation
2. **sequential-thinking** - Structured, step-by-step problem-solving

#### Skills
1. **mcp-installer** - Install/manage MCP servers
2. **git-commit-helper** - Generate commit messages
3. **skill-creator** - Create/update Claude skills
4. **pdf-processing-pro** / **pdf-anthropic** - PDF processing
5. **mcp-builder** - Build MCP servers

#### Subagents (PROACTIVE)
1. **project-context-analyzer** - MUST BE USED to understand current project state before implementing changes
2. **coder** - MUST BE USED for ALL code implementation, modification, debugging (Phases 1/2/3)
3. **github** - MUST BE USED for ALL Git/GitHub operations (branching, commits, PRs, repo inspection)

### Automatic Tool Selection Rules

#### Sequential Thinking MCP - Use For:
- **Complex System Analysis** (3+ components/steps): RL environments, training pipelines, multi-file workflows
- **Multi-Step Problem Solving**: Debugging complex issues, root cause analysis, performance bottlenecks
- **Design & Planning**: New features, implementation approaches, architectural decisions
- **Code Investigation**: Understanding code before modification, refactoring strategies
- **Decision-Making with Trade-offs**: Comparing approaches, evaluating choices

**Trigger Keywords**: "study", "analyze", "investigate", "determine", "understand", "how does", "why is", "debug", "plan", "design", "compare", "evaluate"

#### Context7 MCP - Use For:
- **Working with External Libraries**: Stable Baselines3, PyTorch, TensorFlow, Pandas, NumPy, Gymnasium
- **Best Practices & Modern Techniques**: "latest", "up-to-date", "modern", "current", "recommended"
- **Library Version Specifics**: Version-specific features, avoiding deprecated APIs
- **Documentation-Heavy Tasks**: Complex library features, framework conventions

**Trigger Keywords**: Library names, "best practices", "latest", "documentation"

#### Combined Usage (Sequential Thinking + Context7):
Use BOTH when task is complex AND involves external libraries (e.g., "Analyze training pipeline and refactor using best practices")

#### Project-Context-Analyzer Subagent - Use For:
**CRITICAL: ALWAYS use BEFORE implementing features, debugging, or making architectural decisions**
- **Session Initialization**: Read changelog.md at start of new chat sessions (MANDATORY per workflow)
- **Pre-Implementation Analysis**: Understand current code structure before modifications
- **Feature Location**: Find where specific functionality is implemented
- **Current State Assessment**: Identify recent changes, ongoing work, known issues
- **Cross-File Investigation**: Trace logic across multiple files (environments, training scripts, data pipeline)
- **Apex Compliance Review**: Locate and verify compliance rule enforcement
- **Configuration Discovery**: Find current hyperparameters, settings, and configurations
- **Refactoring Preparation**: Map full scope of changes needed across codebase
- **Debugging Context**: Investigate current implementation before fixing bugs
- **Documentation Alignment**: Cross-check code against docs and CLAUDE.md

**Trigger Keywords**: "where", "current", "how is X implemented", "before we", "understand", "locate", "find", "what's the current", "recent changes"

**IMPORTANT**:
- Invoke BEFORE coder subagent when making changes to existing code
- Automatically reads changelog.md at start (per project workflow)
- Provides file:line_number references for easy navigation
- Read-only (never modifies code - delegates to coder subagent)
- Works in tandem with coder and github subagents

**Typical Workflow**:
```
User request → Project-Context-Analyzer: Analyze current state →
Coder: Implement based on context → Github: Commit + push
```

#### Coder Subagent - Use For:
**CRITICAL: ALWAYS use for ANY code implementation, modification, or debugging**
- Feature implementation (all phases)
- Bug fixes
- Code refactoring
- Code modification (hyperparameters, reward functions, action/observation spaces)
- Debugging & error resolution

**Trigger Keywords**: "implement", "fix", "bug", "error", "refactor", "modify", "update", "change", "write code", "add feature", "debug", "optimize"

**IMPORTANT**: Invoke BEFORE attempting any code changes. Do NOT modify code directly - always delegate to coder subagent.

#### Github Subagent - Use For:
**CRITICAL: ALWAYS use for ALL repository management**
- **Branch Management**: Create simple, descriptive branches (NEVER work on main directly)
- **Commit Operations**: Atomic commits with proper 7-rule format + Claude footers
- **Pull Request Workflow**: Create PRs, review, merge (ONLY when user confirms), squash, delete branches
- **Repository Inspection**: git status, diffs, commit history
- **Changelog Enforcement**: Verify changelog.md updated for major changes

**Git Best Practices (MANDATORY)**:
1. **Simple Branch Names**: `phase2`, `sltp-fix`, `jax-migration`, `rewards` (NOT feature/*, bugfix/*)
2. **Atomic Commits**: One logical change per commit
3. **Commit Message Format**:
   - Subject: ≤50 chars, imperative mood
   - Blank line
   - Body: WHY and HOW (wrap at 72 chars)
   - Required footer: "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
   - Required co-author: "Co-Authored-By: Claude <noreply@anthropic.com>"

**Workflow Example**:
```
Session start → Project-Context-Analyzer: Read changelog.md →
User request → Project-Context-Analyzer: Analyze current state → Github: Create branch →
Coder: Implement → Github: Check changelog → Github: Commit + push →
User confirms → Github: Create PR + merge → Github: Delete branch
```

### Skills - Trigger Phrases
- **mcp-installer**: "install MCP server", "configure MCP", "manage MCP servers"
- **git-commit-helper**: "create a commit", "write commit message"
- **skill-creator**: "create a new skill", "update skill", "build a skill"
- **pdf-***: "process PDF", "extract from PDF", "merge PDFs", "OCR PDF"
- **mcp-builder**: "build an MCP server", "create MCP integration"

### Decision Matrix - Quick Reference

| Task Type | Project-Context | Sequential | Context7 | Coder | Github | Notes |
|-----------|-----------------|------------|----------|-------|--------|-------|
| Session start | ✅ | ❌ | ❌ | ❌ | ❌ | Read changelog.md first |
| Locate feature | ✅ | ❌ | ❌ | ❌ | ❌ | Find implementation |
| Understand current state | ✅ | ⚠️ | ❌ | ❌ | ❌ | Get project context |
| Pre-implementation review | ✅ | ⚠️ | ⚠️ | ❌ | ❌ | Before making changes |
| Implement feature | ✅ | ⚠️ | ⚠️ | ✅ | ✅ | Context→Coder→Github |
| Fix bug | ✅ | ❌ | ❌ | ✅ | ✅ | Context→Coder→Github |
| Refactor | ✅ | ⚠️ | ⚠️ | ✅ | ✅ | Context→Think→Coder |
| Modify code | ✅ | ❌ | ❌ | ✅ | ✅ | Context→Coder→Github |
| Debug complex | ✅ | ✅ | ⚠️ | ✅ | ❌ | Context→Think→Coder |
| Plan | ⚠️ | ✅ | ⚠️ | ❌ | ❌ | Planning only |
| Study system | ✅ | ✅ | ❌ | ❌ | ❌ | Deep understanding |
| Git operations | ❌ | ❌ | ❌ | ❌ | ✅ | All Git/GitHub tasks |

**Legend**: ✅ ALWAYS | ⚠️ Conditional | ❌ Not needed

### Usage Guidelines

**DO**:
- **ALWAYS use project-context-analyzer at session start** to read changelog.md
- **ALWAYS use project-context-analyzer BEFORE code changes** to understand current state
- **ALWAYS delegate code work to coder subagent**
- **ALWAYS delegate Git operations to github subagent**
- **ALWAYS create branches for new work** (simple names)
- **ALWAYS update changelog.md for major changes**
- Use project-context-analyzer to locate features and understand implementations
- Use Sequential Thinking for non-trivial tasks
- Use simple branch names (JAX, rewards, bug-fix)
- Wait for user confirmation before merging

**DON'T**:
- **NEVER skip project-context-analyzer at session start** - read changelog.md first
- **NEVER modify code without understanding current state** - use project-context-analyzer first
- **NEVER modify code directly** - use coder subagent
- **NEVER work on main branch** - create branch first
- **NEVER merge without confirmation**
- **NEVER skip changelog updates**
- **NEVER use long branch names** or feature/*, bugfix/* prefixes
- Skip Git workflow for "quick fixes"
- Make assumptions about current code structure without verification
- Create vague commit messages

## Architecture Overview

### Three-Phase Curriculum Learning

```
Phase 1: Entry Signal Learning (2M timesteps, ~6-8 hours)
├─ Action Space: 3 (Hold, Buy, Sell)
├─ Focus: Entry signal quality
├─ SL/TP: Fixed (1.5x ATR SL, 3:1 ratio)
└─ Output: phase1_foundational_final.zip + vecnorm

Phase 2: Position Management (5M timesteps, ~8-10 hours)
├─ Action Space: 6 (Entry/Exit + SL/TP management)
├─ Transfer Learning: Auto-loads newest Phase 1 model
├─ Focus: Risk management & position optimization
├─ SL/TP: Dynamic (learnable)
└─ Output: phase2_position_mgmt_final.zip + vecnorm

Phase 3: Hybrid RL + LLM (5M timesteps, ~12-16 hours)
├─ Action Space: 6 (same as Phase 2)
├─ LLM: FinGPT LoRA (Meta-Llama-3-8B, 4-bit quantized)
├─ Decision Fusion: Confidence-weighted voting
├─ Hardware: GPU with 8GB+ VRAM
└─ Output: Hybrid agent for deployment
```

### Observation Space Evolution

| Phase | Dimensions | Key Features |
|-------|-----------|--------------|
| Phase 1 | 225 | 11 base indicators × 20 window + 5 position state |
| Phase 2 | 228 | 225 + 3 validity flags (action masking) |
| Phase 3 | 261 | 228 + 33 LLM context features |

### Action Space

**Phase 1**: Hold, Buy, Sell
**Phase 2 & 3**: Hold, Buy, Sell, Move SL to Break-Even, Enable Trailing Stop, Disable Trailing Stop

## Project Structure

```
AI Trainer/
├── main.py                          # MAIN ENTRY POINT (Interactive CLI)
├── requirements.txt                 # Dependencies
├── changelog.md                     # CRITICAL: Update after major changes
├── CLAUDE.md                        # This file
├── config/llm_config.yaml          # Phase 3 LLM config
├── src/                            # Core source (29 files)
│   ├── train_phase*.py             # Training scripts (1/2/3)
│   ├── environment_phase*.py       # Trading environments
│   ├── evaluate_phase*.py          # Evaluation
│   ├── update_training_data.py     # Data pipeline
│   ├── market_specs.py             # 8 futures specifications
│   ├── apex_compliance_checker.py  # Compliance validation
│   └── [other modules]
├── data/                           # Training data (CSV)
├── models/                         # Saved checkpoints
├── tests/                          # Test suite (8 files)
└── docs/                          # Documentation
    ├── Apex-Rules.md               # Compliance requirements
    ├── HYBRID_ARCHITECTURE.md      # Phase 3 architecture
    └── [other docs]
```

## File Organization Guidelines

**CRITICAL**: Maintain strict folder organization. NEVER create files in the root directory unless they are core project files.

### File Location Rules

#### Root Directory (ONLY)
**Allowed files**: `main.py`, `requirements.txt`, `changelog.md`, `CLAUDE.md`, `README.md`, `setup.py`, `.gitignore`, configuration files (`.env`, `pyproject.toml`)

**Forbidden**: Analysis files, reports, plans, summaries, temporary files, test files

#### Documentation Files → `docs/`
ALL documentation, analysis, plans, reports, and summaries MUST be saved to:
```
/home/javlo/Code Projects/RL Trainner & Executor System/AI Trainer/docs/
```

**Examples**:
- Analysis reports: `docs/IMPLEMENTATION_REPORT.md`
- Implementation plans: `docs/MIGRATION_PLAN.md`
- Feature summaries: `docs/PHASE1_IMPROVEMENTS_SUMMARY.md`
- Stress test reports: `docs/JAX_STRESS_TEST_REPORT.md`
- Architecture docs: `docs/HYBRID_ARCHITECTURE.md`
- Checklists: `docs/IMPLEMENTATION_CHECKLIST.md`

#### Test Files → `tests/`
ALL test Python files MUST be saved to:
```
/home/javlo/Code Projects/RL Trainner & Executor System/AI Trainer/tests/
```

**Examples**:
- Unit tests: `tests/test_environment.py`
- Integration tests: `tests/test_llm_integration.py`
- Benchmark scripts: `tests/benchmark_features.py`
- Test analysis: `tests/analyze_test_results.py`
- Smoke tests: `tests/test_menu_smoke.py`

#### Source Code → `src/`
Core implementation code only:
```
/home/javlo/Code Projects/RL Trainner & Executor System/AI Trainer/src/
```

#### Temporary/Utility Scripts → `scripts/`
One-off scripts, utilities, stress tests:
```
/home/javlo/Code Projects/RL Trainner & Executor System/AI Trainer/scripts/
```

**Examples**:
- `scripts/stress_hardware_jax.py`
- `scripts/run_hybrid_test.py`
- `scripts/test_phase1_improvements.sh`

### Automatic Organization Workflow

When creating new files, Claude MUST:
1. **Identify file type** (documentation, test, source, utility)
2. **Use correct target directory** from rules above
3. **NEVER save to root** unless it's a core project file
4. **Move misplaced files** to correct locations when discovered

### Examples

❌ **WRONG**:
```python
# Creating file at root
Write("/home/javlo/.../AI Trainer/IMPLEMENTATION_PLAN.md")
Write("/home/javlo/.../AI Trainer/test_new_feature.py")
```

✅ **CORRECT**:
```python
# Documentation to docs/
Write("/home/javlo/.../AI Trainer/docs/IMPLEMENTATION_PLAN.md")

# Tests to tests/
Write("/home/javlo/.../AI Trainer/tests/test_new_feature.py")
```

## Key Concepts

### 1. Curriculum Learning
Progressive complexity: Phase 1 (entry signals) → Phase 2 (position management) → Phase 3 (LLM reasoning)

### 2. Transfer Learning
Phase 2 auto-loads newest Phase 1 model to inherit entry signal knowledge

### 3. Multi-Market Support
8 futures markets (ES, NQ, YM, RTY, MNQ, MES, M2K, MYM) defined in `market_specs.py`

### 4. Action Masking (Phase 2)
`MaskablePPO` prevents invalid actions (e.g., selling when not in position)

### 5. Apex Compliance (Three Layers)
1. Environment: Rules in reward function + done signal
2. Wrapper: Safety validation before action execution
3. Verification: Post-training compliance checks

### 6. Hybrid Decision Fusion (Phase 3)
RL + LLM via confidence-weighted voting with risk veto

## Data Processing Pipeline

**Input**: Raw GLBX .zip files from Databento (minute/second OHLCV)

**Process**:
1. Extract & detect OHLC format
2. Corruption detection (IQR, percentile analysis)
3. Auto-fix (multiply corrupted bars by 100)
4. Timezone conversion (US Eastern)
5. Filter trading hours (9:30 AM - 4:00 PM ET)
6. Calculate 11+ technical indicators
7. Add market features (regime detection, volatility)
8. Validate quality (IQR checks)
9. Output: {MARKET}_D1M.csv & {MARKET}_D1S.csv

**Expected Medians (Nov 2025)**: ES: $6,400 | NQ: $24,000 | YM: $44,000 | RTY: $2,200

**Key Files**: `update_training_data.py`, `process_new_data.py`, `reprocess_from_source.py`, `data_validator.py`

## Technical Indicators

**Base (all phases)**: SMA (5,20,50,200), RSI (14,15,60), MACD (12,26,9), ATR (14), Bollinger Bands (20,2), Stochastic, Williams %R, CCI, ADX, ROC, MFI

**Phase 3 Extended**: ADX slope, VWAP distance, multi-timeframe SMA, pattern recognition, support/resistance, double top/bottom, breakout signals

## LLM Configuration (Phase 3)

- **Model**: Phi-3-mini-4k-instruct
- **Quantization**: INT4 (bnb 4-bit, ~6GB VRAM)
- **LLM Weight**: 0.15 (15% trust)
- **Confidence Threshold**: 0.75
- **Query Interval**: 5 bars
- **Risk Veto**: Max consecutive losses = 3, Min win rate = 0.4

**IMPORTANT**: Phase 3 requires Meta-Llama-3-8B weights in `fingpt-mt_llama3-8b_lora/Base_Model/` + FinGPT adapter files. CLI auto-downloads base if `HF_TOKEN` set.

## Common Workflows

### Training
```bash
# Phase 1
python src/train_phase1.py              # Production (2M timesteps)
python src/train_phase1.py --test       # Test mode

# Phase 2 (auto-loads Phase 1)
python src/train_phase2.py
python src/train_phase2.py --test

# Phase 3 (requires LLM)
python src/train_phase3_llm.py
python src/train_phase3_llm.py --test   # 30 min test
```

### Data Processing
```bash
python src/process_new_data.py --market NQ          # New data
python src/update_training_data.py --market ES      # Main pipeline
python src/reprocess_from_source.py --market NQ     # Fix corruption
```

### Continue Training
```bash
python main.py  # Select: Training Model → Continue from Existing Model
# OR
python src/train_phase1.py --continue --model-path models/phase1_*.zip
```

### Evaluation
```bash
python src/evaluate_phase2.py
python src/evaluate_phase3_llm.py --model models/phase3_hybrid_final --market NQ
```

### Testing
```bash
python -m pytest tests/ -v                           # All tests
python -m pytest tests/test_llm_integration.py -v    # Specific
```

## Important Constraints

### Apex Trader Funding Compliance (CRITICAL)
1. $2,500 max trailing drawdown
2. All trades close by 4:59 PM ET
3. No overnight positions
4. No daily loss limit
5. 7+ trading days for evaluation
6. Position size: 0.5-1.0 contracts

**See**: `docs/Apex-Rules.md`

### Model Management
- VecNormalize state: Always save/load with model
- Phase 2: Auto-loads newest Phase 1 model
- Checkpointing: .zip files with metadata
- Continue training: Preserves timestep count

### Thread Pool Management
```python
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
```

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Apex Compliance | 100% | Enforced |
| Sharpe Ratio | > 2.5 | Phase 2 goal |
| Win Rate | > 50% | Target |
| Max Drawdown | < 5% | Enforced |

## Hardware Requirements

**Minimum**: CPU 8+ cores, RAM 16GB, Disk 20GB, GPU optional (Phase 1/2)
**Recommended (Phase 3)**: RTX 3060+ (8GB+ VRAM), RAM 32GB, Disk 50GB, CUDA 11.8+

## Common Issues

**CUDA OOM**: Reduce batch_size (256) or num_envs (40)
**Training Divergence**: Reduce learning rate (1e-4), increase clip range (0.3)
**Import Errors**: Run `python setup.py install`
**Model Not Found (Phase 2)**: Ensure Phase 1 completed, model in `models/` with "phase1" in filename
**Data Corruption**: Run `python src/reprocess_from_source.py --market NQ`

## Development Guidelines

### Adding New Features
1. Update relevant `src/` files
2. Add tests in `tests/`
3. Update docs in `docs/`
4. **Update changelog.md** (CRITICAL)
5. Ensure Apex compliance

### Changelog Workflow (CRITICAL)

**IMPORTANT**: changelog.md is the central record of all project changes. Ensures continuity across chat sessions.

#### When to Update
Update `changelog.md` immediately after:
- Major code changes (features, bug fixes, refactoring, optimizations)
- Configuration changes (hyperparameters, environment, LLM)
- Documentation updates
- Dependency changes

#### Starting New Chat Session (MANDATORY)
1. **Read changelog.md FIRST** (last 3 days if too large)
2. Review recent changes
3. Identify ongoing work/issues
4. Use context to inform decisions

#### Changelog Entry Format
```markdown
## [Date: YYYY-MM-DD] - Session Title

### Added
- New features/capabilities

### Changed
- Modifications to functionality

### Fixed
- Bug fixes

### Removed
- Deprecated features

### Notes
- Important context/decisions
- Known issues/future work
```

#### Best Practices
- Be specific (file names, function names, line numbers)
- Explain WHY (not just what)
- Link related work
- Flag breaking changes with ⚠️
- Update immediately (don't batch)

**Development Flow**: Read changelog → Understand context → Implement → Test → Update changelog → Commit

### Code Style
- Follow existing patterns
- Use type hints
- Add docstrings for public functions
- Keep functions modular

### Testing
- Run `pytest tests/ -v` before committing
- Add tests for new features
- Ensure Apex compliance tests pass

## Important File References

### Project Management
- `changelog.md:1` - **CRITICAL: Read at start of every session** - Complete history
- `CLAUDE.md:1` - This file

### Entry Points
- `main.py:1` - Interactive CLI (MAIN ENTRY POINT)

### Core Training
- `src/train_phase1.py:125` - Phase 1 config
- `src/train_phase2.py:140` - Phase 2 config
- `src/train_phase3_llm.py:1` - Hybrid LLM training

### Environments
- `src/environment_phase1.py:1` - Base trading environment
- `src/environment_phase2.py:1` - Position management
- `src/environment_phase3_llm.py:1` - LLM-enhanced

### Data & Compliance
- `src/update_training_data.py:1` - Main pipeline
- `src/data_validator.py:1` - Validation (detect_and_fix_price_format)
- `src/market_specs.py:1` - MarketSpecification dataclass
- `src/apex_compliance_checker.py:1` - Compliance validation

### Documentation
- `docs/Apex-Rules.md:1` - Complete compliance rules
- `docs/HYBRID_ARCHITECTURE.md:1` - Phase 3 architecture
- `docs/LLM_INTEGRATION_GUIDE.md:1` - LLM setup

## Version History

**v1.0.0** (October 2025)
- Three-phase curriculum learning
- Multi-market support (8 futures)
- Apex compliance enforcement
- LLM integration (Phase 3)
- Automatic corruption detection
- Continue-from-checkpoint
- Action space reduction (9 → 6)

## Quick Reference Commands

```bash
# Setup
python main.py                                       # Interactive menu
pip install -r requirements.txt                      # Manual install

# Data
python src/process_new_data.py --market NQ           # Process new
python src/reprocess_from_source.py --market NQ      # Fix corruption

# Training
python src/train_phase1.py --test                    # Test Phase 1
python src/train_phase2.py                           # Production Phase 2
python src/train_phase3_llm.py --test                # Test Phase 3

# Continue
python src/train_phase1.py --continue --model-path models/phase1_*.zip

# Evaluation
python src/evaluate_phase2.py                        # Evaluate Phase 2

# Testing
python -m pytest tests/ -v                           # All tests

# Monitoring
tensorboard --logdir tensorboard_logs/               # TensorBoard
```

## Contact & Support

**Author**: Javier | **X**: [@javiertradess](https://x.com/javiertradess)

**For Issues**: Check `logs/`, review `docs/Apex-Rules.md`, run `pytest tests/ -v`, review `docs/FIXES_SUMMARY.md`

---

**Note**: System designed for trading education, research, and automated execution. Always maintain strict Apex compliance and risk management.
