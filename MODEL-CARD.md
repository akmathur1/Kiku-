# Model Card: Kiku

Kiku is Molterra's speech recognition module: a multiclass neural network for
automatic speech recognition (ASR) and speech translation, implemented from
scratch in Rust. This card follows the structure of Model Cards for Model
Reporting (Mitchell et al., 2018). It states what the model is, how it is
trained, where it is known to fail, and how Molterra turns its output into
live reasoning and durable memory while a meeting is still happening.

## Model Details

Kiku is a sequence to sequence encoder decoder Transformer. Audio goes in as
an 80 channel log Mel spectrogram computed over 30 second windows (25 ms
analysis window, 10 ms hop, 16 kHz mono input). The encoder is a stack of
pre activation Transformer blocks over a two layer convolutional stem, with
sinusoidal position information. The decoder is an autoregressive Transformer
with learned positions, self attention, and cross attention into the encoder
output, and its output projection is tied to the token embedding.

The multiclass structure is what lets one network replace several stages of a
classical speech pipeline. The decoder's vocabulary contains, beyond the text
tokens, classes for every supported language, for the task (transcribe or
translate), for voice activity (`<|nospeech|>`), and for time (a timestamp
class every 20 ms). Language identification, task selection, voice activity
detection, segmentation, and transcription are all predictions over the same
softmax, made in a fixed grammar: start token, language class, task class,
then timestamped text until the end token.

The runtime loads checkpoints in the open safetensors layout. Several sizes
of the same architecture are supported.

| size | parameters | notes |
|---|---|---|
| tiny | 39 M | development and smoke default |
| base | 74 M | |
| small | 244 M | |
| medium | 769 M | |
| large | 1550 M | best accuracy |

English only variants of the smaller sizes exist and tend to do better on
English only audio. Pruned decoder variants trade translation ability for
speed and should not be used when translation into English is needed.

### Model type

Sequence to sequence ASR and speech translation, with joint language
identification, voice activity detection, and timestamp prediction.

### Where the code lives

The production inference runtime is the Rust crate in this repository. The
research side, the full creation of the network, lives in `notebooks/`:
data preparation, the audio frontend, tokenizer training, the architecture,
the training loop, decoding, evaluation, and checkpoint export.

## Training

The training recipe, written out end to end in the notebooks, is large scale
weak supervision: hundreds of thousands of hours of audio paired with
transcripts of uneven quality, filtered mechanically rather than hand
labeled.

**Data preparation.** Source corpora are scanned into a single manifest
format: utterance id, audio path, transcript, language, task, duration,
speaker, and an optional English translation. Filtering removes utterances
that are too short or too long, transcripts whose length is implausible for
their audio duration (an alignment proxy that catches mismatched pairs),
and near duplicate transcripts that would let the model memorize instead of
listen. Audio is resampled to 16 kHz mono, peak normalized, and trimmed of
leading and trailing silence. Splits are made by speaker, never by
utterance, so a voice heard in training is never scored in test.

**Frontend and augmentation.** Training consumes the same 80 channel log Mel
representation the runtime computes, so there is no train versus serve skew
in the feature space. During training the waveform is randomly gained,
mixed with noise, speed perturbed, and occasionally reverberated; the
spectrogram is then masked in time and frequency (SpecAugment). Evaluation
uses the clean frontend only.

**Text and targets.** Transcripts are tokenized with a byte level BPE
vocabulary trained on the transcript text itself, so no character sequence
is ever out of vocabulary. Each training target is the full multitask
sequence: start, language class, task class, then either plain text or
timestamped text, then end. Timestamped and plain targets are mixed within
training so the model learns both forms. For translation pairs the audio is
non English and the target text is English, under the translate task class.

**Optimization.** Training minimizes cross entropy over the target sequence
with teacher forcing, ignoring the prompt positions so the loss is only on
what the model must produce. The optimizer is AdamW with decoupled weight
decay applied to matrix weights only, betas 0.9 and 0.98, linear warmup
into a cosine decay schedule, and global gradient norm clipping at 1.0.
Training runs in mixed precision with gradient accumulation to reach large
effective batch sizes. Checkpoints carry the optimizer and random number
generator state so a run resumes exactly where it stopped.

**Evaluation during and after training.** Held out loss is tracked during
the run. Final accuracy is measured as word error rate after text
normalization, per language, with character error rate for languages
written without spaces (zh, ja, th, lo, my, km). Error analysis buckets
utterances into exact, near, degraded, and failed, and surfaces the worst
cases for inspection.

Performance in any given language tracks the amount of training audio in
that language. High resource languages reach strong error rates; low
resource languages can be an order of magnitude worse, and some are only
usable with the larger sizes. Kiku's FLEURS harness (`eval_fleurs`) measures
this per language on our own runtime, and the LibriSpeech harness
(`eval_librispeech`) tracks English.

## Not Simple Transcription

A plain transcriber maps audio to a string. Kiku's architecture is chosen
so that it registers transcription instead: every prediction leaves a
measurable trace that downstream systems can hold, cite, and doubt.

The difference is structural, not a post processing step:

- **The output is a register, not a string.** Each segment carries its start
  and end times (from the timestamp classes), the identified language (from
  the language classes), the average log probability of its text, and the
  no speech probability of its window. A word without a time, a language,
  and a confidence does not leave the decoder.
- **Uncertainty is a first class output.** Because voice activity and
  confidence come from the same softmax as the text, the model can say
  "something was said here and I am not sure what" instead of inventing a
  fluent sentence. A plain transcriber has no channel for that statement.
- **Time is predicted, not aligned afterwards.** The timestamp classes make
  segmentation part of decoding, so every span of text is addressable back
  to a span of audio. That addressability is what makes a transcript
  citable evidence rather than prose.
- **One model, several verdicts.** Language identification, task selection,
  voice activity, and segmentation are all classes of the one output layer,
  so the evidence about a segment is internally consistent: the same
  forward pass that produced the words produced the doubt about them.

## Pairing with the Memory Layer

This register is the contract between hearing and memory. Molterra's memory
layer stores durable, cited organizational memory, and it only accepts what
it can later defend:

- **Provenance.** A fact absorbed from a meeting points back to the segment
  that grounds it: which audio span, which language, at what confidence.
  Memory without a pointer back to evidence is not written.
- **The evidence gate.** Confident segments can enter trusted reasoning and
  become graph facts, notes, and tasks. Weak segments remain display
  material: a human can read them, the memory layer does not learn from
  them.
- **Closed lexicon repair.** A mangled name is repaired only against the
  tenant's own lexicon of people, companies, and terms already in memory.
  The memory layer constrains the transcript; the transcript never invents
  entries in the memory layer.
- **Degradation instead of fabrication.** When audio is weak, the register
  records uncertainty and memory abstains. The failure mode is a gap that
  can be asked about, never a confident falsehood that gets remembered.

A transcript in this design is not the product. It is the evidence stream
that lets meeting speech become organizational memory without the memory
ever trusting a guess.

## From Register to Real Time Output

The register is not written to disk and read later. It is consumed as it is
produced, and the point of producing it in this form is that logic can run on
it while the people in the room are still talking. What follows is the shape
of that path as Molterra runs it; the implementation of the reasoning and
memory stages lives in Molterra, not in this repository.

**Streaming.** The model is trained on 30 second windows, but the runtime
does not wait 30 seconds to speak. Audio is decoded over a rolling window
with a cached decoder state, so a segment is registered within moments of
its end timestamp, and a segment already registered is never silently
rewritten by a later window. Retained session audio is heard a second time
after the meeting in a record pass; the live and record registers are
reconciled into one canonical transcript, and where they disagree the
disagreement is kept as evidence rather than resolved by picking a winner.

**Signal extraction.** Each registered segment is read for the things a
meeting turns on: the names of people, companies, and products; acronyms
and jargon; numbers, dates, and amounts; questions, and commitments of the
form someone will do something by some time. Every extracted signal keeps
the span and confidence of the segment it came from. Numbers are a hard
boundary: no repair stage may rewrite a number, because a confidently
wrong amount is worse than a marked uncertain one. Names are repaired only
against the tenant's closed lexicon, and only when the acoustic evidence
for the word is strong enough that the repair is a correction rather than
a wish.

**Deciding what to say.** The signals feed a battery of deterministic
triggers, evaluated on every window against the tenant's memory: a question
asked about something memory knows; a person or company named who has a
history with this team; a figure stated that memory holds a different value
for; a commitment made that should become a task. Each trigger produces a
candidate with a grounding score, that score is calibrated to a probability,
and a candidate surfaces only above a threshold derived from a declared
cost of interrupting versus a declared cost of staying quiet. Thresholds
are not tuned by hand. Every evaluation is logged, including the ones that
surface nothing, so silence can be measured as precisely as speech.

**Output the user can act on.** What reaches the screen in real time is
grounded by construction: one fact, one source, the source being a memory
entry with its own provenance or the audio span that grounds the claim. A
question asked in the room can be answered from connected systems while the
room is still on that topic; if nothing in the workspace grounds an answer,
the answer is labeled as general knowledge and never dressed as workspace
fact. A commitment becomes a drafted task only when its recipient,
deliverable, and context are each supported by evidence; otherwise it is
held as a commitment without a draft. Weak audio produces less output, not
worse output.

**Why the multitask design is what makes this possible.** Each of those
stages needs the same four things about a span of text: when it was said,
in what language, how confident the model was, and whether it was speech at
all. A string transcriber would need those reconstructed after the fact by
separate models, each with its own errors and none of them consistent with
the words. Because Kiku predicts them jointly, the evidence a trigger fires
on is the evidence the words were decoded with, and the real time logic can
trust that a low confidence span is low confidence about the very words it
is looking at.

## Intended Use in Molterra

Kiku exists to turn meeting speech into evidence carrying text. Molterra's
hearing pipeline gates on the register described above: a confident segment
can enter trusted reasoning, a weak one is display material only, and
repairs are closed over the tenant's own lexicon so nothing is invented.

Recording in Molterra is consent gated. We do not use Kiku, and recommend
against using any ASR model, to transcribe people recorded without their
consent, to classify speakers, or to drive high risk decisions directly from
raw transcripts. Transcripts are inputs to a gated pipeline, not ground
truth.

Kiku never decides who is speaking. A sequence to sequence decoder will
guess speaker names from transcript context; that guess is discarded.
Speaker identity in Molterra comes from structural channel attribution and
meeting scoped diarization in the capture pipeline.

## Performance and Limitations

The architecture is robust to accents, background noise, and technical
vocabulary relative to classical pipelines, and translates from many
languages into English zero shot. It is not perfect, and the failure modes
are specific:

- **No word accuracy guarantee.** State of the art is low error, not zero
  error. The evidence fields exist so downstream consumers gate rather than
  trust.
- **Hallucination.** Weakly supervised sequence to sequence models can emit
  text that was never spoken, most often on non speech audio, long silence,
  or heavy noise, because the decoder is part language model. Kiku counters
  this at decode time: a window is only dropped as non speech when the no
  speech probability exceeds 0.6 and the average log probability is below
  -1, and low confidence text never enters trusted reasoning.
- **Repetition loops.** Autoregressive decoding can loop. Kiku detects this
  with a compression ratio test (zlib ratio above 2.4) and retries the
  window at higher temperature, keeping the best evidenced attempt.
- **Uneven performance across languages and speakers.** Accuracy differs by
  language, accent, acoustic conditions, and demographic dimensions of the
  speaker. Hallucination and repetition are worse in lower resource
  languages. Molterra measures rather than assumes: the per language
  harnesses exist for exactly this.
- **Latency is a property of the runtime, not the model.** The model is
  trained on 30 second windows; live behavior comes from rolling window
  decoding with a cached decoder state and from the streaming, reconciling
  capture pipeline above the model. A segment's confidence is available at
  the moment the segment is, which is what lets the real time logic gate
  on it, but the first words of a window are known later than a purely
  frame synchronous system would report them.

## Broader Implications

Cheap accurate transcription cuts both ways. It improves accessibility and
makes meeting memory possible, and it also lowers the cost of surveillance.
Molterra's position is structural, not aspirational: recording requires
consent, transcripts carry provenance and confidence, retention is bounded
and owned, and no subjective classification of speakers is performed or
exposed. An ASR model's output here is evidence to be gated, never an
authority.
