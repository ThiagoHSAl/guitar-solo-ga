# Gerador de solos de guitarra por algoritmo genético

Trabalho Prático 1 de DCC831 – IA Generativa para Música (UFMG, 2026/2).
Autor: Thiago Henrique Silva de Almeida (matrícula 2024014180).

O sistema gera solos de guitarra em formato simbólico (MIDI) usando um **algoritmo
genético**. O solo é tocado sobre uma base de acompanhamento (bateria, baixo e harmonia)
que o próprio sistema gera a partir de uma progressão de acordes.

- **Restrição estilística:** um solo sobre uma progressão de acordes fixa em um gênero
  popular. O caso principal é o blues de 12 compassos na levada blues shuffle. Há 19
  levadas de 10 gêneros (blues, rock, hard rock, metal, punk, indie, reggae, country,
  funk e pop).
- **Indivíduo:** uma matriz de inteiros sobre uma grade rítmica, com quatro linhas:
  melodia (nota, pausa ou sustentação), nota dupla, articulação (bend, deixar soar) e a
  origem da nota (qual lick a escreveu).
- **Fitness:** heurística, com 10 critérios de teoria musical e fraseado calculados em
  relação aos acordes da base: harmonia, consonância, intervalos, densidade, shred,
  coerência rítmica, fraseado, motivo, final e idioma.
- **Licks:** uma biblioteca de 25 licks (frases curtas típicas de cada estilo) é usada na
  população inicial e em uma das mutações.

O resumo no formato ISMIR está em [`resumo/resumo.pdf`](resumo/resumo.pdf).

## Músicas geradas

A pasta [`musicas/`](musicas/) tem cinco peças. Cada uma traz o áudio (`completo.mp3`),
os arquivos MIDI (`solo.mid`, `base.mid`, `bateria.mid`, `completo.mid`), o gráfico da
fitness (`fitness.png`) e os parâmetros da execução (`run.json`).

| Pasta | Gênero | Tonalidade e escala do solo | Andamento |
|---|---|---|---|
| `01_blues_shuffle_A_seed3` | blues shuffle (12 compassos) | Lá, escala blues com terça maior | 100 bpm |
| `02_thrash_Em_seed1` | thrash metal | Mi menor, modo frígio | 180 bpm |
| `03_hard_rock_Em_seed4` | hard rock | Mi menor, modo dórico | 126 bpm |
| `04_funk_Em_seed1` | funk | Mi menor, modo dórico | 96 bpm |
| `05_balada_C_seed1` | balada pop | Dó maior, escala maior | 72 bpm |

## Instalação

Requisitos: Python 3.12 (testado em Linux) e a biblioteca do FluidSynth, que transforma
o MIDI em áudio.

```bash
# 1. FluidSynth (sistema)
sudo apt install fluidsynth          # Debian/Ubuntu
# brew install fluid-synth           # macOS

# 2. Ambiente Python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Soundfont (timbres).** O áudio usa o soundfont GeneralUser GS, que não vai no
repositório por causa do tamanho. Baixe em
<https://schristiancollins.com/generaluser.php> e salve o arquivo `.sf2` na pasta
`soundfonts/` (por exemplo, `soundfonts/GeneralUser-GS.sf2`). Também funciona qualquer
soundfont General MIDI instalado no sistema (ex.: `sudo apt install fluid-soundfont-gm`)
ou indicado pela variável de ambiente `SOUNDFONT`. Sem soundfont, os arquivos MIDI são
gerados normalmente; só o áudio fica de fora.

## Uso

### Interface gráfica

```bash
streamlit run app.py
```

A interface tem duas etapas: primeiro se escolhe e se ouve a base (progressão,
tonalidade, levada e andamento); depois se gera o solo, acompanhando a evolução da
fitness. Todos os parâmetros do algoritmo genético ficam em "Avançado".

### Linha de comando

```bash
python cli.py list                  # levadas, escalas, progressões e critérios da fitness

# só a base (bateria, baixo e harmonia)
python cli.py backing --preset "12-bar blues" --tonic A --rhythm shuffle

# solo completo
python cli.py solo --preset "12-bar blues" --tonic A --rhythm shuffle --seed 3
```

Cada execução de `solo` cria uma pasta em `outputs/` com os MIDI, o áudio, o gráfico da
fitness e o `run.json`. Principais opções de `solo` (`python cli.py solo -h` mostra
todas):

| Opção | Padrão | Significado |
|---|---|---|
| `--seed` | 42 | semente aleatória; a mesma configuração com outra semente gera outro solo |
| `--pop` | 120 | tamanho da população |
| `--gens` | 300 | número máximo de gerações |
| `--patience` | 80 | para se o melhor indivíduo não melhora por esse número de gerações |
| `--mutation` | 0,3 | probabilidade de cada uma das até quatro mutações por filho |
| `--crossover` | 0,9 | probabilidade de cruzamento |
| `--tournament` | 3 | tamanho do torneio de seleção |
| `--elitism` | 2 | indivíduos copiados sem alteração para a geração seguinte |
| `--immigrants` | 0,05 | fração da população trocada por indivíduos novos a cada geração |
| `--init` | `walk` | população inicial por passeio aleatório na escala (`walk`) ou uniforme |
| `--lick-rate` | 0,3 | chance de um lick por compasso na população inicial (0 = sem licks) |
| `--scale` | `auto` | escala do solo; `auto` usa a escala padrão da levada |
| `--weight` | – | muda o peso de um critério, ex.: `--weight harmony=4` |

Também é possível passar uma progressão própria com `--chords`, um acorde por compasso
(ex.: `--chords "Em7 A7 Em7 A7" --mode minor --tonic E`).

## Reprodução dos resultados

**Músicas.** Os comandos abaixo recriam a pasta `musicas/`.

Toda a aleatoriedade do sistema vem de um único gerador de números pseudoaleatórios,
iniciado com o valor de `--seed` (a semente). Isso inclui a população inicial, a
seleção, o cruzamento, as mutações, os imigrantes e os detalhes de expressão do MIDI
(variação de intensidade das notas e escolha dos bends). Por isso:

**a mesma configuração com outra semente gera uma música diferente (como solicitado pelo enunciado)**: a busca parte de
  outra população e segue outro caminho, chegando a outro solo com fitness parecida. Por
  exemplo, trocar `--seed 3` por `--seed 4` no blues gera outro solo sobre a mesma base.
  Nos experimentos do resumo, cinco sementes no blues deram cinco solos diferentes, com
  fitness entre 0,96 e 0,97.

- **a mesma configuração com a mesma semente gera exatamente a mesma música**, nota por
  nota: os arquivos MIDI saem idênticos byte a byte. É isso que torna os experimentos
  reprodutíveis, e os comandos abaixo devolvem as mesmas peças entregues;

O áudio pode variar levemente se for usado outro soundfont ou outra versão do
FluidSynth; o MIDI não depende deles.

```bash
python cli.py solo --preset "12-bar blues" --tonic A --rhythm shuffle --seed 3 \
    --out musicas --name 01_blues_shuffle_A_seed3
python cli.py solo --preset "i-VI-VII (metal)" --tonic E --rhythm thrash --seed 1 \
    --out musicas --name 02_thrash_Em_seed1
python cli.py solo --preset "i-VII-VI-VII (rock menor)" --tonic E --rhythm hard_rock --seed 4 \
    --out musicas --name 03_hard_rock_Em_seed4
python cli.py solo --chords "Em7 A7 Em7 A7 Em7 A7 Em7 A7" --tonic E --mode minor \
    --rhythm funk --seed 1 --out musicas --name 04_funk_Em_seed1
python cli.py solo --preset "I-vi-IV-V (indie / doo-wop)" --tonic C --rhythm ballad \
    --scale ionian --seed 1 --out musicas --name 05_balada_C_seed1
```

A CLI grava o áudio em WAV (`completo.wav`). Os MP3 do repositório são esses mesmos
áudios convertidos para ocupar menos espaço.

**Experimentos do resumo.** As 35 execuções da Tabela 2 e a Figura 1 (variação de
semente, taxa de mutação, imigrantes, inicialização, licks e gênero) são reproduzidas
por:

```bash
python resumo/experimentos.py . resumo/exp          # roda em paralelo; alguns minutos
python resumo/figura_fitness.py resumo/exp/results.json resumo/fitness.pdf
```

Os resultados usados no resumo estão em `resumo/resultados.json`.

**Resumo.** O PDF é compilado a partir de `resumo/resumo.tex` (template oficial do
ISMIR 2026), por exemplo com `tectonic resumo/resumo.tex` ou com
`pdflatex` + `bibtex` + `pdflatex` duas vezes.

## Estrutura

```
app.py                interface Streamlit
cli.py                linha de comando
solo_ga/
  theory.py           escalas, acordes e progressões prontas
  rhythms.py          as 19 levadas (bateria, baixo, acompanhamento) e perfis de solo
  backing.py          geração da base
  licks.py            biblioteca de 25 licks
  midi_io.py          conversão para MIDI (bends, vibrato, nota dupla)
  render.py           síntese de áudio com FluidSynth
  plots.py            gráfico da fitness
  ga/
    genome.py         representação do indivíduo e população inicial
    fitness.py        os 10 critérios da fitness
    operators.py      seleção, cruzamento e mutações
    engine.py         laço do algoritmo genético
musicas/              as cinco peças entregues
resumo/               resumo ISMIR (LaTeX e PDF), experimentos e figura
```

## Uso de IA

Foram usadas ferramentas de IA (Claude, da Anthropic):

- **Planejamento do projeto** (escolha do método, representação e critérios):
  conversa no claude.ai – https://claude.ai/chat/d1f4d4ea-3d53-4638-b7f7-656c1771e320
  
- **Implementação, experimentos e resumo**, com o Claude Code: a partir das
  especificações e decisões do autor, a IA rodou os experimentos e revisou/editou o resumo. O autor definiu o
  problema, a restrição estilística, os gêneros, as levadas e os critérios desejados. Ele
  também ouviu as músicas geradas e dirigiu as mudanças feitas pela IA a partir delas: menos pausas e
  notas picotadas, trinado de três notas, troca dos gêneros das músicas, escolha das
  sementes, inclusão dos imigrantes, teste e remoção do critério de groove.
