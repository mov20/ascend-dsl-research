# Альтернативные языки для Huawei Ascend

**Статус:** 🟢 Активный
**Старт:** 2026-03-30
**Горизонт:** 3–12 месяцев
**Владелец:** Oleg (рабочий проект)

## Суть

Исследование и разработка темы альтернативных языков программирования для платформы Huawei Ascend (NPU/AI-ускорители).

## Контекст

- Huawei Ascend — линейка AI-процессоров (Ascend 310, 910 и др.)
- Основной стек: MindSpore, CANN (Compute Architecture for Neural Networks), AscendCL
- **Фокус: разработка собственного Python DSL** для программирования Ascend NPU
- **Цель 1 — Простота:** минимум строк кода, понятный синтаксис, низкий порог входа
- **Цель 2 — Производительность:** ≥90% от пикового потенциала железа Ascend

## Архитектура

- **Уровень абстракции:** Triton-like kernel DSL (operator/kernel level)
- **Roadmap:** позже — выход на graph-level для оптимизации полных pipeline (fusion, scheduling)
- **Компиляция:** Python DSL → MLIR → AscendC code generation
- **Целевое железо:** Ascend NPU (Cube Unit для матричных операций, Vector Unit для поэлементных)

## Языки для анализа

| Язык | Кто | Фокус |
|------|-----|-------|
| **Triton** | OpenAI | Kernel DSL, Python, MLIR, GPU |
| **Triton-Ascend** | Huawei | Форк Triton под Ascend NPU — **ключевой референс/конкурент** |
| **TileLang** | tile-ai (open-source) | Tile-based абстракции, GPU |
| **TileLang-Ascend** | tile-ai (open-source) | TileLang для Ascend NPU (A2/A3), AscendC codegen — **ключевой референс/конкурент** |
| **Pallas** | Google/JAX | Kernel DSL для TPU/GPU через JAX |
| **Gluon** | ? | TBD |
| **cuTile** | NVIDIA | Tile-уровневые абстракции для CUDA |
| **Mojo** | Modular | Python-superset, systems-level perf |
| **AscendCraft** | Исследовательский | LLM-driven DSL → AscendC автогенерация ядер (arxiv 2601.22760) |
| **Helion** | Meta/PyTorch | Python DSL, tile-based, поверх Triton |

**Цель анализа:** синтаксические конструкции, абстракции, подходы к достижению near-peak performance. Что берём, что нет, почему.

## Открытые вопросы

- [ ] Есть ли доступ к железу Ascend?
- [ ] Целевая аудитория DSL? (ML-инженеры, kernel-разработчики?)
- [x] IR-стратегия: **MLIR** (выбрано)
- [ ] Лицензирование / open-source?

## Process Rules

1. **Plan first** — always write an action plan and get approval before executing
2. **Batch work** — avoid one-by-one token-heavy iterations; group tasks
3. **Documents in English**
4. **Trigger phrases:**

## Commands

- **«Саммари по Ascend»** — получить резюме всех обсуждений и решений
- **«Саммари по Ascend за [период]»** — резюме за конкретный промежуток
- **«Новости по Ascend»** — дайджест свежих новостей по индустрии (Huawei Ascend, CANN, MindSpore, альт. языки для NPU)
- **«Новости по Ascend за [период]»** — новости за конкретный промежуток

## Лог

### 2026-03-30
- Трек заведён
- Определён фокус: Python DSL → MLIR → AscendC codegen
- Цели: простота (мин. LOC) + ≥90% peak performance
- Уровень: Triton-like kernel DSL, с заделом на graph-level
- Составлена сводная таблица 7 языков: `memory/projects/ascend-dsl-comparison.md`
- Добавлены Ascend-специфичные проекты: Triton-Ascend, TileLang-Ascend, AscendCraft
- TODO: глубокий анализ AscendCraft (DSL-дизайн, host/kernel разделение, UB/L1 буферы)
