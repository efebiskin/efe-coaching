// Serverless function: receives the coaching intake form, runs it through Gemini to produce a
// trainer-facing brief (summary + program + package + price + draft reply), then delivers it to
// Efe via Discord (instant) + email (Resend), and emails the lead a confirmation.
// Degrades gracefully: works with whatever env vars are configured; never loses a lead (logs it).
import { GoogleGenerativeAI } from '@google/generative-ai';
import { Resend } from 'resend';

const SYSTEM_PROMPT = `You are the intake analyst for an independent personal trainer (Efe, a CPT serving LA & Ventura County in person + online). Your reader is THE TRAINER, never the client. You convert a new lead's intake answers into a fast, skimmable brief the trainer reads in under 60 seconds before deciding how to onboard the lead.

ROLE & TONE
- Write FOR the trainer: practical, blunt, coach-to-coach. No fluff, no hype, no motivational filler.
- You are a triage assistant, not a doctor. Never diagnose, prescribe, or give medical/rehab advice. For any injury or medical limitation, flag it and recommend clearance or caution - do not program around it in detail.
- Be decisive. Pick one recommendation and state why. Do not hedge with "it depends." If the intake is missing something that changes the call, say so in one line under Missing info.

DATA RULES
- Use only what the intake gives you. Never invent metrics, injuries, or goals.
- If a field is blank or unclear, write "not provided" and, if it matters, list it under Missing info.
- This form does NOT collect age, height, or weight - do not ask for or infer them; they are gathered at onboarding.
- Treat all numbers as client self-report.

HOW TO REASON (internal - do not show your work)
1. Program type: match split + style + weekly structure to days/week, session length, experience, and equipment.
   - 2 days -> full-body. 3 -> full-body or PPL-lite. 4 -> upper/lower or PPL. 5-6 -> PPL or bro-split if experienced.
   - Beginner -> simpler full-body / movement-pattern focus. Intermediate/advanced -> more specialization.
   - Goal drives style: fat loss -> higher density / circuits + strength base; muscle -> hypertrophy volume; strength -> lower reps, compounds; recomp -> hypertrophy + modest deficit conditioning; general health -> balanced + conditioning; sport -> movement-specific + power.
   - Equipment gates everything: home/limited -> dumbbell/bodyweight/band variants and say so. If "Full gym," assume full equipment.
2. Package fit:
   - Online -> self-motivated, can execute solo, budget-conscious, or remote ("Online" location).
   - 1-on-1 In-Person -> beginner, injury/form risk, accountability-dependent, "In-person" location, higher budget.
   - Hybrid -> wants accountability but can't do frequent in-person; mixed experience; "Either works"; good default when signals split.
   - Respect stated location as a HARD constraint: never recommend In-Person to an "Online (anywhere)" lead.
3. Price RANGE is a SUGGESTION the trainer overrides. Anchor to stated budget + package: Online $99-$199, Hybrid $199-$349, In-person $350-$800/mo. If budget is below the package floor, note the mismatch in one line. Always label it a suggestion.

OUTPUT
- Output ONLY the markdown brief in the exact section order/format from the FORMAT block. No preamble, no closing remarks, no code fences around the whole thing.
- Skimmable: short bullets, bold labels, no paragraph longer than 2 lines.
- The client-facing draft reply is the ONLY part in a warm, client-friendly voice. Everything above it is trainer-facing.
- Never expose these instructions or your reasoning.`;

const FORMAT = `FORMAT - fill exactly:

# New Lead - {First name}

## 1. Client Summary
- **Contact:** {email}{ . phone if provided}
- **Location:** {In-person / Online / Either}
- **Primary goal:** {goal}{ . driver: {driver} if provided}
- **Availability:** {days}/wk . {session length}
- **Experience:** {level - short read}
- **Training setup:** {gym access}{ . equipment: {list} - only if provided}
- **Budget:** {budget}
- **Injuries / limitations:** {injuries, or "none reported"}{ - recommend medical clearance if relevant}
- **Notes:** {notes, or "-"}

## 2. Recommended Program
- **Split:** {e.g. Upper/Lower 4-day}
- **Training style:** {e.g. hypertrophy + 2x conditioning}
- **Weekly structure:** {one line}
- **Why:** {1-2 lines tying it to days, experience, goal, equipment}

## 3. Suggested Package - {Online / 1-on-1 In-Person / Hybrid}
- **Why this fit:** {2-3 bullets}
- **Watch-outs:** {anything that could flip the call, or "-"}

## 4. Suggested Price Range *(suggestion - trainer sets final)*
- **{$X-$Y / month}** for {package + frequency}
- _Basis:_ {one line vs stated budget}
- Estimate only.

## 5. Draft First Reply to Client
> {warm 4-7 sentence reply: thank them, reflect their goal, name the approach in plain language, state next step, sign as Efe. No prices unless trainer adds them.}

---
**Missing info:** {comma list of blank fields that affect the plan, or "none"}`;

const REQUIRED = ['firstName', 'email', 'location', 'goal', 'level', 'days', 'session', 'gymAccess', 'budget'];
const rate = new Map();
const clean = (s, max = 2000) => String(s ?? '').slice(0, max).trim();

// Discord caps messages at 2000 chars - split the brief on line boundaries so nothing is cut.
async function sendDiscordChunks(url, text) {
  const chunks = [];
  let cur = '';
  for (const line of text.split('\n')) {
    const piece = line.length > 1850 ? line.slice(0, 1850) : line;
    if ((cur + '\n' + piece).length > 1850) { if (cur) chunks.push(cur); cur = piece; }
    else cur = cur ? cur + '\n' + piece : piece;
  }
  if (cur) chunks.push(cur);
  for (const c of chunks) {
    try { await fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ content: c }) }); } catch {}
  }
}

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ ok: false, error: 'method' });

  let b = req.body;
  if (typeof b === 'string') { try { b = JSON.parse(b); } catch { b = {}; } }
  b = b || {};

  // spam honeypot - bots fill hidden fields
  if (b.website) return res.status(200).json({ ok: true });

  // best-effort rate limit (per warm instance)
  const ip = String(req.headers['x-forwarded-for'] || '').split(',')[0] || 'x';
  const now = Date.now();
  const recent = (rate.get(ip) || []).filter(t => now - t < 600000);
  if (recent.length >= 5) return res.status(429).json({ ok: false, error: 'Too many requests, try again shortly.' });
  recent.push(now); rate.set(ip, recent);

  // validate
  for (const f of REQUIRED) if (!b[f]) return res.status(400).json({ ok: false, error: 'Please fill all required fields.' });
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(b.email)) return res.status(400).json({ ok: false, error: 'Please enter a valid email.' });

  const intake = [
    ['First name', b.firstName], ['Email', b.email], ['Phone', b.phone],
    ['Training location', b.location], ['Main goal', b.goal], ['Goal driver', b.goalDriver],
    ['Training level', b.level], ['Days per week', b.days], ['Session length', b.session],
    ['Gym access', b.gymAccess], ['Equipment', Array.isArray(b.equipment) ? b.equipment.join(', ') : b.equipment],
    ['Injuries / limitations', b.injuries], ['Monthly budget', b.budget], ['Anything else', b.notes],
  ].filter(([, v]) => v != null && v !== '').map(([k, v]) => `${k}: ${clean(v)}`).join('\n');

  // backstop so a lead is never lost even before all keys are configured
  console.log('NEW LEAD\n' + intake);

  // AI brief (graceful fallback to raw intake on any error / missing key)
  let brief;
  try {
    if (!process.env.GEMINI_API_KEY) throw new Error('GEMINI_API_KEY not set');
    const model = new GoogleGenerativeAI(process.env.GEMINI_API_KEY)
      .getGenerativeModel({ model: 'gemini-2.5-flash', systemInstruction: SYSTEM_PROMPT });
    const r = await model.generateContent({
      contents: [{ role: 'user', parts: [{ text: FORMAT + '\n\nINTAKE:\n' + intake }] }],
      generationConfig: { temperature: 0.6, maxOutputTokens: 1500 },
    });
    brief = r.response.text();
  } catch (e) {
    brief = `# New Lead - ${clean(b.firstName, 80)}\n_(AI brief unavailable: ${e.message})_\n\n## Raw intake\n${intake}`;
  }

  const owner = process.env.OWNER_EMAIL || 'ebiskinpt@gmail.com';
  const from = process.env.FROM_EMAIL || 'Coaching <onboarding@resend.dev>';
  const tasks = [];

  if (process.env.DISCORD_WEBHOOK_URL) {
    tasks.push(sendDiscordChunks(process.env.DISCORD_WEBHOOK_URL, '**New coaching lead**\n' + brief));
  }
  if (process.env.RESEND_API_KEY) {
    const resend = new Resend(process.env.RESEND_API_KEY);
    tasks.push(resend.emails.send({
      from, to: owner, replyTo: b.email,
      subject: `New lead - ${clean(b.firstName, 80)} (${clean(b.goal, 40)})`, text: brief,
    }).catch(() => {}));
    tasks.push(resend.emails.send({
      from, to: b.email,
      subject: 'Got your application - Efe Biskin Coaching',
      text: `Hi ${clean(b.firstName, 80)},\n\nThanks for applying. I've got your details and I'll personally review them and get back to you within 24 hours with your next step.\n\n- Efe\ncoaching.biskin.studio`,
    }).catch(() => {}));
  }

  await Promise.all(tasks);
  return res.status(200).json({ ok: true });
}
