# Claude.md - RL Trainer AI Trading System

## Project Overview

**RL Trainer** is a production-ready reinforcement learning trading system designed to develop automated trading strategies for futures markets while maintaining strict compliance with Apex Trader Funding rules. This project uses a three-phase curriculum learning approach combining PPO (Proximal Policy Optimization) with optional LLM integration for context-aware trading.

**Version**: 1.0.0 (October 2025)
**Author**: Javier ([@javiertradess](https://x.com/javiertradess))
**License**: Apache License 2.0 (inherited from TensorTrade)
**Python**: 3.11+ (3.13 supported)

**⚠️ CRITICAL WORKFLOW NOTE**: At the start of EVERY new chat session, Claude MUST read [changelog.md](changelog.md) first to understand recent changes and project context. After completing any major changes, update the changelog immediately. See the "Changelog Workflow" section in Development Guidelines for complete details.

## Core Purpose

Train AI agents to trade 8 futures markets (ES, NQ, YM, RTY, MNQ, MES, M2K, MYM) profitably while maintaining **100% Apex Trader Funding compliance**:
- $2,500 max trailing drawdown
- 4:59 PM ET mandatory position close
- No overnight positions
- No daily loss limit
- 7+ trading days minimum for evaluation

## Technology Stack

- **RL Framework**: Stable Baselines3 (PPO, MaskablePPO)
- **Environment**: Gymnasium (OpenAI gym successor)
- **Deep Learning**: PyTorch 2.0+, TensorFlow 2.15+
- **Trading Library**: TensorTrade (custom fork)
- **Data Processing**: Pandas, NumPy, SciPy, Scikit-Learn
- **LLM (Phase 3)**: FinGPT LoRA on Meta-Llama-3-8B
- **Monitoring**: TensorBoard, Custom callbacks

## Intelligent Tool Selection Framework

**CRITICAL**: This section defines when Claude should automatically use MCP servers and Skills based on task context WITHOUT explicit user invocation. The goal is intelligent, context-aware tool usage that enhances productivity.

### Available Tools

#### MCP Servers (Model Context Protocol)
1. **context7** - Provides up-to-date documentation for libraries and frameworks
2. **sequential-thinking** - Enables structured, step-by-step problem-solving

#### Skills (Specialized Capabilities)
1. **mcp-installer** - Install and manage MCP servers
2. **git-commit-helper** - Generate descriptive commit messages
3. **skill-creator** - Create and update Claude skills
4. **pdf-processing-pro** - Production PDF processing workflows
5. **pdf-anthropic** (user) - PDF manipulation toolkit
6. **mcp-builder** (user) - Build MCP servers

### Automatic Tool Selection Rules

#### Sequential Thinking MCP - Use AUTOMATICALLY For:

**ALWAYS use Sequential Thinking when the task involves:**

1. **Complex System Analysis** (3+ components/steps)
   - Analyzing RL environments, training pipelines, or architecture
   - Understanding multi-file workflows or data processing pipelines
   - Investigating how Phase 1/2/3 systems work
   - Studying codebase structure and dependencies

2. **Multi-Step Problem Solving**
   - Debugging complex issues with multiple potential causes
   - Determining root causes of training failures or errors
   - Analyzing performance bottlenecks

3. **Design & Planning Tasks**
   - Designing new features or system components
   - Planning implementation approaches
   - Determining best practices for complex scenarios
   - Architecting solutions with multiple considerations

4. **Code Investigation & Refactoring**
   - Understanding existing code before modification
   - Planning refactoring strategies
   - Analyzing technical debt and improvement opportunities

5. **Decision-Making with Trade-offs**
   - Comparing multiple implementation approaches
   - Evaluating architectural choices
   - Selecting optimal algorithms or data structures

**Trigger Keywords:** "study", "analyze", "investigate", "determine", "understand", "how does", "why is", "debug", "plan", "design", "compare", "evaluate", "best approach", "figure out"

**Example Prompts:**
- "Study the RL environment and determine the best implementation"
- "Analyze why training is diverging in Phase 2"
- "Investigate how the reward function works"
- "Design a new feature for position management"
- "Determine the best way to handle multi-market training"

#### Context7 MCP - Use AUTOMATICALLY When:

**Use Context7 when the task involves:**

1. **Working with External Libraries**
   - Implementing features using Stable Baselines3, PyTorch, TensorFlow, Pandas, NumPy, Gymnasium
   - Referencing API methods or library-specific patterns
   - Using version-specific features or avoiding deprecated APIs

2. **Best Practices & Modern Techniques**
   - User asks for "best practices", "latest", "up-to-date", "modern", "current"
   - Implementing recommended patterns for specific libraries
   - Following official documentation guidelines

3. **Library Version Specifics**
   - Working with specific versions (e.g., "PyTorch 2.0+", "Stable Baselines3 2.x")
   - Avoiding outdated API usage
   - Using newly-introduced features

4. **Documentation-Heavy Tasks**
   - Implementing complex library features
   - Configuring library-specific settings
   - Following framework conventions

**Trigger Keywords:** Library names (Stable Baselines3, PyTorch, TensorFlow, Gymnasium, Pandas, NumPy), "best practices", "latest", "up-to-date", "modern", "current", "how to use", "documentation", "recommended"

**Example Prompts:**
- "Implement a custom callback using the latest Stable Baselines3 API"
- "Use PyTorch best practices for the neural network"
- "What's the modern way to configure Gymnasium environments?"
- "Implement transfer learning with current TensorFlow patterns"

#### Combined Usage (Sequential Thinking + Context7):

**Use BOTH when the task is complex AND involves external libraries:**

**Trigger Conditions:**
- Complex multi-step task + library/framework usage
- Design/planning + need for up-to-date documentation
- System analysis + implementation with best practices

**Example Prompts:**
- "Study the RL environment and determine the best implementation based on latest Stable Baselines3 documentation"
- "Design a new reward function using modern PyTorch patterns"
- "Analyze the training pipeline and refactor using best practices"
- "Investigate Phase 3 LLM integration and implement improvements with latest FinGPT patterns"

### Skill Automatic Usage Rules

Skills should be invoked automatically when user requests EXPLICITLY match skill capabilities:

#### mcp-installer Skill
**Trigger Phrases:**
- "install MCP server"
- "configure MCP"
- "manage MCP servers"
- "add MCP integration"
- "set up [service] MCP"

#### git-commit-helper Skill
**Trigger Phrases:**
- "create a commit"
- "write commit message"
- "commit these changes"
- "generate commit message"
- "review staged changes"

#### skill-creator Skill
**Trigger Phrases:**
- "create a new skill"
- "update the [name] skill"
- "build a skill"
- "package the skill"

#### pdf-processing-pro / pdf-anthropic Skills
**Trigger Phrases:**
- "process PDF", "extract from PDF", "fill PDF form"
- "merge PDFs", "split PDF", "extract tables"
- "OCR this PDF"

#### mcp-builder Skill
**Trigger Phrases:**
- "build an MCP server"
- "create MCP integration"
- "develop MCP server"

### Decision Matrix - Quick Reference

| Task Type | Sequential Thinking | Context7 | Skill | Notes |
|-----------|---------------------|----------|-------|-------|
| Code analysis | ✅ ALWAYS | ❌ | ❌ | Structured investigation |
| Debug complex issue | ✅ ALWAYS | ⚠️ If libraries involved | ❌ | Systematic problem-solving |
| Plan implementation | ✅ ALWAYS | ⚠️ If using libraries | ❌ | Design with reasoning |
| Implement with library | ⚠️ If complex | ✅ ALWAYS | ❌ | Need current docs |
| Simple bug fix | ❌ | ❌ | ❌ | Direct fix |
| Install MCP | ❌ | ❌ | ✅ mcp-installer | Explicit tool request |
| Create commit | ❌ | ❌ | ✅ git-commit-helper | Git operation |
| Process PDF | ❌ | ❌ | ✅ pdf-* | PDF-specific task |
| Create skill | ❌ | ❌ | ✅ skill-creator | Skill management |
| Study system | ✅ ALWAYS | ❌ | ❌ | Understanding codebase |
| Refactor code | ✅ ALWAYS | ⚠️ If libraries | ❌ | Planning + analysis |
| Best practices | ⚠️ If complex | ✅ ALWAYS | ❌ | Need current patterns |

### Implementation Examples

#### Example 1: Complex Analysis Task
**User Prompt:** "Please study the RL environment and determine the best implementation based on the best documentations"

**Automatic Tool Selection:**
- ✅ **Sequential Thinking**: Multi-step analysis of environment structure
- ✅ **Context7**: Fetch latest Gymnasium and Stable Baselines3 documentation
- ❌ **Skills**: Not applicable

**Reasoning:**
- "study" + "determine best" = Complex analysis requiring structured thinking
- "best documentations" = Need up-to-date library docs
- Task involves understanding system architecture AND referencing best practices

#### Example 2: Simple Implementation
**User Prompt:** "Add a print statement showing the current step count"

**Automatic Tool Selection:**
- ❌ **Sequential Thinking**: Simple, single-step task
- ❌ **Context7**: No library-specific patterns needed
- ❌ **Skills**: Not applicable

**Reasoning:** Straightforward code modification, direct implementation.

#### Example 3: Library-Heavy Implementation
**User Prompt:** "Implement a custom PPO callback using Stable Baselines3"

**Automatic Tool Selection:**
- ⚠️ **Sequential Thinking**: Moderate complexity, but not required
- ✅ **Context7**: Need latest Stable Baselines3 callback API
- ❌ **Skills**: Not applicable

**Reasoning:** Library-specific implementation requiring current documentation patterns.

#### Example 4: Complex Design with Best Practices
**User Prompt:** "Design and implement a new reward shaping function using modern PyTorch techniques"

**Automatic Tool Selection:**
- ✅ **Sequential Thinking**: Multi-step design process
- ✅ **Context7**: Latest PyTorch best practices
- ❌ **Skills**: Not applicable

**Reasoning:**
- "Design and implement" = Planning + execution requiring structured approach
- "modern PyTorch techniques" = Need current documentation

#### Example 5: Git Operation
**User Prompt:** "Create a commit for these changes"

**Automatic Tool Selection:**
- ❌ **Sequential Thinking**: Git operation, not analysis
- ❌ **Context7**: Not needed
- ✅ **git-commit-helper Skill**: Explicit git commit request

**Reasoning:** Direct match to git-commit-helper skill capabilities.

### Usage Guidelines for Claude

1. **Proactive Tool Usage**: Use tools automatically based on context WITHOUT waiting for explicit invocation
2. **Sequential Thinking Priority**: When in doubt about complexity, err on the side of using Sequential Thinking
3. **Combined Usage**: Don't hesitate to use multiple tools simultaneously when appropriate
4. **Implicit Detection**: Recognize task patterns from natural language, not just keywords
5. **Skill Precision**: Only invoke skills for exact capability matches (avoid over-triggering)

### Anti-Patterns (What NOT to Do)

❌ **DON'T:**
- Wait for user to say "use sequential thinking" or "use context7"
- Use Context7 for code that doesn't involve external libraries
- Use Sequential Thinking for trivial single-step tasks
- Invoke skills for tasks outside their specific domain
- Ask user which tool to use when the pattern is clear

✅ **DO:**
- Automatically recognize task complexity and select tools
- Use Sequential Thinking liberally for anything non-trivial
- Combine tools when task requires both reasoning and documentation
- Trust the decision matrix for quick reference
- Invoke skills only for explicit, matching requests

## Architecture Overview

### Three-Phase Curriculum Learning

```
Phase 1: Entry Signal Learning (2M timesteps, ~6-8 hours)
├─ Action Space: 3 (Hold, Buy, Sell)
├─ Focus: Entry signal quality
├─ Constraints: Relaxed (learning phase)
├─ SL/TP: Fixed (1.5x ATR SL, 3:1 ratio)
└─ Output: phase1_foundational_final.zip + vecnorm

Phase 2: Position Management (5M timesteps, ~8-10 hours)
├─ Action Space: 6 (Entry/Exit + SL/TP management)
├─ Transfer Learning: Auto-loads newest Phase 1 model
├─ Focus: Risk management & position optimization
├─ Constraints: Strict Apex compliance
├─ SL/TP: Dynamic (learnable)
└─ Output: phase2_position_mgmt_final.zip + vecnorm

Phase 3: Hybrid RL + LLM (5M timesteps, ~12-16 hours)
├─ Action Space: 6 (same as Phase 2)
├─ LLM: FinGPT LoRA (Meta-Llama-3-8B, 4-bit quantized)
├─ Decision Fusion: Confidence-weighted voting
├─ Observation Space: 261D (extended context)
├─ Hardware: GPU with 8GB+ VRAM
└─ Output: Hybrid agent for deployment
```

### Observation Space Evolution

| Phase | Dimensions | Key Features |
|-------|-----------|--------------|
| Phase 1 | 225 | 11 base indicators × 20 window + 5 position state |
| Phase 2 | 228 | 225 + 3 validity flags (action masking) |
| Phase 3 | 261 | 228 + 33 LLM context features (ADX slope, VWAP, patterns) |

### Action Space Details

**Phase 1 (3 actions)**:
- 0: Hold
- 1: Buy (long entry)
- 2: Sell (short entry)

**Phase 2 & 3 (6 actions)** - Simplified from original 9:
- 0: Hold
- 1: Buy (long entry)
- 2: Sell (short entry)
- 3: Move SL to Break-Even
- 4: Enable Trailing Stop
- 5: Disable Trailing Stop

## Project Structure

```
AI Trainer/
├── main.py                          # Interactive CLI menu (MAIN ENTRY POINT)
├── requirements.txt                 # Python dependencies
├── setup.py                         # TensorTrade package setup
├── changelog.md                     # Project changelog (CRITICAL: Update after major changes)
├── CLAUDE.md                        # This file - Project instructions for Claude
│
├── config/
│   └── llm_config.yaml             # Phase 3 LLM configuration
│
├── src/                            # Core source code (29 files)
│   ├── Training & Environments
│   │   ├── train_phase1.py         # Phase 1 training (2M timesteps)
│   │   ├── train_phase2.py         # Phase 2 training (5M timesteps)
│   │   ├── train_phase3_llm.py     # Phase 3 hybrid training
│   │   ├── environment_phase1.py   # Base trading environment
│   │   ├── environment_phase2.py   # Position management environment
│   │   └── environment_phase3_llm.py  # LLM-enhanced environment
│   │
│   ├── Evaluation
│   │   ├── evaluate_phase2.py
│   │   └── evaluate_phase3_llm.py
│   │
│   ├── Data Processing
│   │   ├── update_training_data.py      # Main data pipeline (auto corruption fix)
│   │   ├── process_new_data.py          # Fast processing with corruption detection
│   │   ├── reprocess_from_source.py     # Reprocess corrupted data
│   │   ├── clean_second_data.py
│   │   ├── process_second_data.py
│   │   └── data_validator.py            # Centralized validation functions
│   │
│   ├── Feature Engineering
│   │   ├── feature_engineering.py       # Market regime features
│   │   ├── technical_indicators.py      # 11+ technical indicators
│   │   └── llm_features.py              # Extended LLM features
│   │
│   ├── Trading Logic & Compliance
│   │   ├── market_specs.py              # Market specifications (8 futures)
│   │   ├── apex_compliance_checker.py
│   │   └── llm_reasoning.py
│   │
│   ├── RL & LLM Integration
│   │   ├── hybrid_agent.py              # Decision fusion module
│   │   ├── kl_callback.py               # KL divergence monitoring
│   │   ├── llm_callback.py
│   │   └── model_utils.py               # Model management utilities
│   │
│   └── Utilities
│       ├── metadata_utils.py
│       └── diagnose_environment.py
│
├── data/                           # Training data (CSV files)
├── models/                         # Saved model checkpoints
├── results/                        # Evaluation results
├── logs/                          # Training & execution logs
├── tensorboard_logs/              # TensorBoard monitoring
│
├── tests/                         # Test suite (8 files)
│   ├── test_setup.py
│   ├── test_environment.py
│   ├── test_integration.py
│   ├── test_market_selection.py
│   ├── test_action_masking.py
│   ├── test_llm_minimal.py
│   └── test_llm_integration.py
│
├── docs/                          # Comprehensive documentation
│   ├── Apex-Rules.md                    # Compliance requirements
│   ├── MARKET_SELECTION_IMPLEMENTATION.md
│   ├── QUICK_START_DATA_PROCESSING.md
│   ├── HYBRID_ARCHITECTURE.md           # Phase 3 architecture
│   ├── LLM_INTEGRATION_GUIDE.md
│   ├── NINJATRADER8_INTEGRATION_REQUIREMENTS.md
│   ├── VISUAL_DEMO.md
│   └── FIXES_SUMMARY.md
│
└── .kilocode/rules/
    └── projectrules.md              # Changelog & MCP server rules
```

## Key Concepts & Design Patterns

### 1. Curriculum Learning
The system uses progressive complexity:
- **Phase 1**: Learn entry signals with fixed SL/TP
- **Phase 2**: Learn position management with dynamic SL/TP (inherits Phase 1 weights)
- **Phase 3**: Add LLM reasoning for context-aware decisions

### 2. Transfer Learning
Phase 2 automatically loads the newest Phase 1 model to inherit learned entry signals, focusing training on new position management skills.

### 3. Multi-Market Support
The `MarketSpecification` dataclass in `market_specs.py` defines contract specifications for 8 futures markets:

| Symbol | Name | Multiplier | Tick Value | Default Commission | Type |
|--------|------|-----------|-----------|-------------------|------|
| ES | E-mini S&P 500 | $50 | $12.50 | $2.50/side | E-mini |
| NQ | E-mini Nasdaq-100 | $20 | $5.00 | $2.50/side | E-mini |
| YM | E-mini Dow Jones | $5 | $5.00 | $2.50/side | E-mini |
| RTY | E-mini Russell 2000 | $50 | $5.00 | $2.50/side | E-mini |
| MNQ | Micro Nasdaq | $2 | $0.50 | $0.60/side | Micro |
| MES | Micro S&P 500 | $5 | $1.25 | $0.60/side | Micro |
| M2K | Micro Russell | $5 | $0.50 | $0.60/side | Micro |
| MYM | Micro Dow | $0.50 | $0.50 | $0.60/side | Micro |

### 4. Action Masking (Phase 2)
Uses `MaskablePPO` from `sb3-contrib` to prevent invalid actions (e.g., selling when not in position). Validity features are included in the observation space.

### 5. Apex Compliance (Three-Layer Safety)
1. **Layer 1 (Environment)**: Rules enforced in reward function + done signal
2. **Layer 2 (Wrapper)**: Safety validation before action execution
3. **Layer 3 (Verification)**: Post-training compliance checks

### 6. Hybrid Decision Fusion (Phase 3)
The `HybridTradingAgent` combines RL and LLM recommendations:
- **Agreement**: Both recommend same action → Take it
- **High Confidence**: One very confident → Follow it
- **Disagreement**: Weighted voting based on confidence scores
- **Risk Veto**: Override if risk too high (consecutive losses, near drawdown limit)

## Data Processing Pipeline

### Input
Raw GLBX .zip files from Databento containing minute and second-level OHLCV data.

### Process
1. **Extract & Detect**: Auto-detect OHLC format
2. **Corruption Detection**: Statistical validation using IQR and percentile analysis
3. **Auto-Fix**: Multiply corrupted bars by 100 (divide-by-100 errors)
4. **Timezone Conversion**: Convert to US Eastern Time
5. **Filter Trading Hours**: 9:30 AM - 4:00 PM ET
6. **Calculate Indicators**: 11+ technical indicators
7. **Add Market Features**: Regime detection, volatility analysis
8. **Validate Quality**: Final IQR checks
9. **Generate Output**: {MARKET}_D1M.csv & {MARKET}_D1S.csv

### Corruption Detection Logic
The system compares actual median prices to expected medians for each market:

**Expected Medians (Nov 2025)**:
- ES: $6,400
- NQ: $24,000
- YM: $44,000
- RTY: $2,200
- MES/MNQ/M2K/MYM: Same as E-mini counterparts

**Detection Method**:
- Uses percentile analysis (P1, P5, P95)
- Detects "bimodal distribution" (minority corruption)
- Applies 100× multiplier to corrupted bars only

### Key Data Files
- `update_training_data.py` - Main pipeline with auto corruption fix
- `process_new_data.py` - Fast processing for new data
- `reprocess_from_source.py` - Clean up old corrupted data
- `data_validator.py` - Centralized validation functions

## Technical Indicators (11+)

**Base Indicators** (all phases):
- SMA (5, 20, 50, 200 period)
- RSI (14, 15, 60 period)
- MACD (12, 26, 9)
- ATR (14 period)
- Bollinger Bands (20, 2)
- Stochastic Oscillator
- Williams %R
- CCI (Commodity Channel Index)
- ADX (Average Directional Index)
- ROC (Rate of Change)
- MFI (Money Flow Index)

**Phase 3 Extended Features**:
- ADX slope (trend strength momentum)
- VWAP distance
- Multi-timeframe SMA (50, 200)
- Pattern recognition (higher/lower highs/lows)
- Support/resistance levels
- Double top/bottom detection
- Breakout/breakdown signals

## Configuration Files

### Phase 1 (train_phase1.py:125-135)
```python
PHASE1_CONFIG = {
    'total_timesteps': 2_000_000,
    'num_envs': 80,
    'learning_rate': 3e-4,
    'batch_size': 512,
    'n_steps': 2048,
    'gae_lambda': 0.95,
    'gamma': 0.99,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'device': 'cuda'
}
```

### Phase 2 (train_phase2.py:140-150)
```python
PHASE2_CONFIG = {
    'total_timesteps': 5_000_000,
    'num_envs': 80,
    'learning_rate': 3e-4,
    'batch_size': 512,
    'device': 'cuda'
}
```

### LLM Configuration (config/llm_config.yaml)
- **Base Model**: meta-llama/Meta-Llama-3-8B (download to `fingpt-mt_llama3-8b_lora/Base_Model/`)
- **Adapter**: FinGPT/fingpt-mt_llama3-8b_lora (LoRA files live in `fingpt-mt_llama3-8b_lora/`)
- **Quantization**: INT4 (bnb 4-bit) to hold VRAM near 6GB
- **LLM Weight**: 0.15 (15% trust in LLM decisions)
- **Confidence Threshold**: 0.75
- **Query Interval**: 5 bars (selective querying to reduce latency)
- **Risk Veto**: Max consecutive losses = 3, Min win rate = 0.4

**IMPORTANT**: Before running Phase 3, download the Meta-Llama-3-8B base weights and FinGPT adapter into the folders referenced by `config/llm_config.yaml`. The CLI will refuse to start Phase 3 until both folders exist (it can auto-download the base if `HF_TOKEN` is set).

## Common Workflows

### First-Time Setup
```bash
python main.py
# 1. Requirements Installation
# 2. Data Processing (select market)
# 3. Training Model → Training Pod
# 4. Evaluator
```

### Data Processing
```bash
# NEW data (auto corruption detection)
python src/process_new_data.py --market NQ

# OR main pipeline
python src/update_training_data.py --market ES

# Reprocess old corrupted data
python src/reprocess_from_source.py --market NQ
```

### Training
```bash
# Phase 1 - Entry Learning
python src/train_phase1.py              # Production (2M timesteps)
python src/train_phase1.py --test       # Test mode (reduced dataset)

# Phase 2 - Position Management (auto-loads newest Phase 1)
python src/train_phase2.py
python src/train_phase2.py --test

# Phase 3 - Hybrid RL + LLM (requires Meta-Llama-3-8B + FinGPT LoRA folders)
python src/train_phase3_llm.py
python src/train_phase3_llm.py --test   # 30 min test run
```

**Note**: Phase 3 requires Meta-Llama-3-8B weights inside `fingpt-mt_llama3-8b_lora/Base_Model/` plus the FinGPT adapter files in `fingpt-mt_llama3-8b_lora/`.

### Continue Training from Checkpoint
```bash
# Via menu
python main.py
# Select: Training Model → Continue from Existing Model

# OR command line
python src/train_phase1.py --continue --model-path models/phase1_foundational_final.zip
```

### Evaluation
```bash
python src/evaluate_phase2.py
python src/evaluate_phase3_llm.py --model models/phase3_hybrid_final --market NQ
```

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Specific tests
python -m pytest tests/test_environment.py -v
python -m pytest tests/test_llm_integration.py -v
```

### TensorBoard Monitoring
```bash
tensorboard --logdir tensorboard_logs/
```

## Important Constraints & Rules

### Apex Trader Funding Compliance (CRITICAL)
1. **Trailing Drawdown**: $2,500 max (enforced in environment)
2. **Position Close**: All trades MUST close by 4:59 PM ET
3. **No Overnight Positions**: Automatic close at 4:59 PM
4. **No Daily Loss Limit**: Only trailing drawdown matters
5. **Minimum Trading Days**: 7+ days for evaluation
6. **Position Size**: 0.5-1.0 contracts (configurable per market)

**See**: `docs/Apex-Rules.md` for complete details

### Data Quality Standards
- **No Corruption**: Auto-detect and fix divide-by-100 errors
- **Trading Hours Only**: 9:30 AM - 4:00 PM ET
- **Timezone**: US Eastern Time
- **Validation**: IQR-based outlier detection

### Model Management
- **VecNormalize State**: Always save/load with model
- **Phase 2 Initialization**: Auto-loads newest Phase 1 model
- **Checkpointing**: Models saved as .zip files with metadata
- **Continue Training**: Preserves timestep count and training progress

### Thread Pool Management
The system limits BLAS threads to prevent contention:
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
| GPU Utilization | 85%+ | Optimized |

## Hardware Requirements

**Minimum**:
- CPU: 8+ cores
- RAM: 16GB
- Disk: 20GB
- GPU: Optional for Phase 1/2

**Recommended** (Phase 3 LLM):
- GPU: RTX 3060 or better (8GB+ VRAM)
- RAM: 32GB
- Disk: 50GB
- CUDA: 11.8+

## Common Issues & Solutions

### CUDA Out of Memory
Reduce batch size or number of environments:
```python
'batch_size': 256,  # Down from 512
'num_envs': 40,     # Down from 80
```

### Training Divergence
- Reduce learning rate: `3e-4` → `1e-4`
- Increase clip range: `0.2` → `0.3`
- Check reward scaling

### Import Errors
Ensure TensorTrade is installed:
```bash
python setup.py install
```

### Model Not Found (Phase 2)
The system auto-detects the newest Phase 1 model. Ensure:
- Phase 1 training completed
- Model exists in `models/` directory
- Model filename contains "phase1"

### Data Corruption
Run reprocessing:
```bash
python src/reprocess_from_source.py --market NQ
```

## Development Guidelines

### Adding New Features
1. Update relevant files in `src/`
2. Add tests in `tests/`
3. Update documentation in `docs/`
4. **Update changelog.md** (see Changelog Workflow below)
5. Ensure Apex compliance maintained

### Changelog Workflow (CRITICAL)

**IMPORTANT**: The changelog.md file is the central record of all project changes and development history. Proper changelog management ensures continuity across chat sessions and development cycles.

#### When to Update the Changelog
Update `changelog.md` immediately after completing ANY of the following:

1. **Major Code Changes**:
   - New features or functionality
   - Bug fixes (especially critical ones)
   - Refactoring or architectural changes
   - Performance optimizations
   - Breaking changes

2. **Configuration Changes**:
   - Updates to training hyperparameters
   - Changes to environment configuration
   - Modifications to LLM settings

3. **Documentation Updates**:
   - New documentation files
   - Significant updates to existing docs
   - Changes to this CLAUDE.md file

4. **Dependency Changes**:
   - New dependencies added
   - Version upgrades
   - Package removals

#### Starting a New Chat Session (MANDATORY)

**CRITICAL**: At the beginning of EVERY new chat session, Claude MUST:

1. **Read the changelog.md file FIRST** before starting any work
2. Review recent changes to understand current project state
3. Identify any ongoing work or issues from previous sessions
4. Use the changelog context to inform decisions and recommendations

**Example Workflow**:
```
User: "Let's continue working on the Phase 2 optimization"

Claude Actions:
1. Read changelog.md to understand:
   - What optimizations were already attempted
   - What worked and what didn't
   - Current state of Phase 2 implementation
   - Any known issues or blockers
2. Then proceed with the requested work
```

#### Changelog Entry Format

Each changelog entry should follow this format:

```markdown
## [Date: YYYY-MM-DD] - Session Title

### Added
- New features or capabilities
- New files or modules

### Changed
- Modifications to existing functionality
- Updated configurations or parameters

### Fixed
- Bug fixes
- Resolved issues

### Removed
- Deprecated features
- Deleted files or code

### Notes
- Important context or decisions
- Known issues or future work
- Performance impacts
```

#### Best Practices

1. **Be Specific**: Include file names, function names, and line numbers when relevant
2. **Explain Why**: Document the reasoning behind changes, not just what changed
3. **Link Related Work**: Reference related changes or dependencies
4. **Flag Breaking Changes**: Clearly mark any breaking changes with ⚠️
5. **Update Immediately**: Don't batch changelog updates - add entries as work completes

#### Integration with Development Workflow

```
Development Flow:
1. Start new session → Read changelog.md
2. Understand context from previous work
3. Implement changes
4. Test changes
5. Update changelog.md with details
6. Commit code (changelog.md included)
```

**Anti-Patterns to Avoid**:
- ❌ Forgetting to update changelog after major changes
- ❌ Starting work without reading changelog first
- ❌ Vague changelog entries ("fixed stuff", "made changes")
- ❌ Batching multiple sessions into one changelog entry
- ❌ Omitting important context or reasoning

### Code Style
- Follow existing patterns
- Use type hints where possible
- Add docstrings for public functions
- Keep functions focused and modular

### Testing Requirements
- Run tests before committing: `pytest tests/ -v`
- Add tests for new features
- Ensure Apex compliance tests pass

### MCP Server & Skill Integration (Project Rules)
**See: "Intelligent Tool Selection Framework" section above for comprehensive automatic tool selection guidance.**

This project has context-aware MCP servers and Skills configured:
1. **Sequential Thinking MCP**: Automatically used for complex, multi-step tasks requiring structured reasoning
2. **Context7 MCP**: Automatically used when working with external libraries or needing up-to-date documentation
3. **Skills**: Automatically invoked based on explicit task patterns (git commits, MCP installation, PDF processing, skill creation)

The "Intelligent Tool Selection Framework" section provides detailed rules, decision matrices, and examples for when each tool should be used automatically without explicit user invocation.

## Important File References

### Project Management
- `changelog.md:1` - **CRITICAL: Read at start of every new session** - Complete project history and development log
- `CLAUDE.md:1` - This file - Project instructions and guidelines for Claude

### Entry Points
- `main.py:1` - Interactive CLI menu (MAIN ENTRY POINT)

### Core Training Files
- `src/train_phase1.py:125` - Phase 1 configuration
- `src/train_phase2.py:140` - Phase 2 configuration
- `src/train_phase3_llm.py:1` - Hybrid LLM training

### Environment Definitions
- `src/environment_phase1.py:1` - Base trading environment
- `src/environment_phase2.py:1` - Position management environment
- `src/environment_phase3_llm.py:1` - LLM-enhanced environment

### Data Processing
- `src/update_training_data.py:1` - Main data pipeline
- `src/data_validator.py:1` - Validation functions (detect_and_fix_price_format)
- `src/process_new_data.py:1` - Fast new data processing

### Market Specifications
- `src/market_specs.py:1` - MarketSpecification dataclass

### Model Utilities
- `src/model_utils.py:1` - Model detection, loading, metadata

### Compliance
- `src/apex_compliance_checker.py:1` - Post-training compliance validation

### Documentation
- `docs/Apex-Rules.md:1` - Complete Apex compliance rules
- `docs/QUICK_START_DATA_PROCESSING.md:1` - Data processing guide
- `docs/HYBRID_ARCHITECTURE.md:1` - Phase 3 architecture details
- `docs/LLM_INTEGRATION_GUIDE.md:1` - LLM setup and customization

## Version History

**v1.0.0** (October 2025)
- Three-phase curriculum learning system
- Multi-market support (8 futures)
- Apex compliance enforcement
- LLM integration (Phase 3)
- Automatic corruption detection
- Continue-from-checkpoint feature
- Action space reduction (9 → 6 actions)

## Contact & Support

**Author**: Javier
**X (Twitter)**: [@javiertradess](https://x.com/javiertradess)

**For Issues**:
1. Check logs in `logs/` directory
2. Review `docs/Apex-Rules.md` for compliance questions
3. Run tests: `pytest tests/ -v`
4. Review `docs/FIXES_SUMMARY.md` for recent bug fixes

## Quick Reference Commands

```bash
# Setup
python main.py                                # Interactive menu
pip install -r requirements.txt               # Manual install

# Data
python src/process_new_data.py --market NQ    # Process new data
python src/reprocess_from_source.py --market NQ  # Fix corruption

# Training
python src/train_phase1.py --test             # Test Phase 1
python src/train_phase2.py                    # Production Phase 2
python src/train_phase3_llm.py --test         # Test Phase 3

# Continue
python src/train_phase1.py --continue --model-path models/phase1_*.zip

# Evaluation
python src/evaluate_phase2.py                 # Evaluate Phase 2

# Testing
python -m pytest tests/ -v                    # All tests
python -m pytest tests/test_llm_integration.py -v  # Specific test

# Monitoring
tensorboard --logdir tensorboard_logs/        # TensorBoard
```

---

**Note**: This system is designed for trading education, research, and potential automated execution. Always maintain strict Apex compliance and risk management practices.
