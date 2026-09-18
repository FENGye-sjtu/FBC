"""Step-boundary execution on a real VLM. Every result records implementation tier."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from efa.core import Budget,Claim,Ledger,Policy,resolve
from efa.qwen import QwenBackend,parse_json
from efa.regions import Box
from efa.format import step_prompt, STEP_END


def run(backend,question,policies):
    ledger=Ledger(); budget=Budget(); trace=[]; final=None
    for step in range(1,budget.max_steps+1):
        budget.take_step()
        exposed=tuple(ledger.live)
        context=[dict(key=k,claim=asdict(n.claim),verified=n.effective_verified) for k,n in ledger.live.items()]
        backend.premises=context
        prompt=step_prompt(question,context,step,step==budget.max_steps)
        raw=backend.generate(prompt,[Box(0,0,backend.image.width,backend.image.height)],max_tokens=384,step_boundary=True)[0]
        try:
            if not raw.rstrip().endswith(STEP_END):
                raise ValueError("Missing step boundary")
            parsed=parse_json(raw); atoms=parsed["atoms"]
            if not 1<=len(atoms)<=budget.max_atoms_per_step or not isinstance(parsed.get("final"),bool):
                raise ValueError("Invalid step coverage")
            claims=[Claim(a["subject"],a["predicate"],a["value"],a.get("family","attribute"),
                          a.get("kind","visual"),exposed) for a in atoms]
        except (ValueError,KeyError,TypeError) as error:
            trace.append(dict(step=step,raw=raw,reason="parse_failure",local_pass=False))
            final=dict(answer=raw,verified=False,reason="parse_failure")
            break
        resolved=[]
        for i,claim in enumerate(claims):
            policy=policies.get(claim.family,policies["default"])
            resolution=resolve(claim,backend,policy,budget)
            sources=[e["source"] for e in resolution.events if e.get("source")]
            source_id=sources[-1]["version"] if sources else "unavailable"
            node=ledger.append(f"step{step}.atom{i}",resolution.claim,resolution.local_pass,exposed,source_id)
            resolved.append(dict(resolution=asdict(resolution),node=asdict(node)))
        # Reconstruct only from audited atoms: no unchecked free-form rewriting.
        text=f"Step {step}: "+"; ".join(f"{x['resolution']['claim']['subject']} / {x['resolution']['claim']['predicate']}: {x['resolution']['claim']['value']}" for x in resolved)
        trace.append(dict(step=step,text=text,atoms=resolved,exposed_parents=exposed))
        if parsed["final"] or step==budget.max_steps:
            final=dict(answer=resolved[-1]["resolution"]["claim"]["value"],
                verified=all(n.effective_verified for n in ledger.live.values()),
                reason="final_audited" if parsed["final"] else "global_step_budget")
            if not parsed["final"]:
                final["verified"]=False
            break
    return dict(question=question,final=final,trace=trace,budget=asdict(budget),cost=backend.calls,
                implementation=backend.representation,calibrated=False)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--image",required=True)
    p.add_argument("--question",required=True);p.add_argument("--policy",required=True);p.add_argument("--output",required=True)
    p.add_argument("--quantized",action="store_true")
    a=p.parse_args(); config=json.loads(Path(a.policy).read_text())
    policies={k:Policy(**v) for k,v in config["strata"].items()}
    backend=QwenBackend(a.model,a.image,a.quantized)
    result=run(backend,a.question,policies)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result["final"]))
