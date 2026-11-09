"""Read-only integrity and checkpoint replay audit for a completed E1/E2 run."""
import argparse
import json
from pathlib import Path
import torch
from tsba.classifier import ParameterMap, ResidualClassifier
from tsba.score import ScoreNetwork
from .archive import digest, write_json


def verify(root):
    manifest=json.loads((root/'manifest.json').read_text())
    for relative,record in manifest.items():
        path=root/relative
        assert path.stat().st_size==record['bytes'] and digest(path)==record['sha256'],relative
    checkpoints=0;rounds=0;score_checkpoints=0;paired=0
    for directory in sorted(root.glob('d*_seed*')):
        if not (directory/'cell_summary.json').exists():continue
        inputs=torch.load(directory/'inputs.pt',weights_only=True)
        source=inputs['source'];dim=source.shape[1]
        model=ResidualClassifier(dim,2,min(dim,16),dtype=source.dtype).eval()
        model.load_state_dict(inputs['initial_head'])
        params=ParameterMap(model);initial=params.vector()
        with torch.no_grad():
            torch.testing.assert_close(model(inputs['evaluation']),torch.load(directory/'initial_predictions.pt',weights_only=True),rtol=0,atol=0)
        reference=None
        for arm in ('learned','empirical','population','zero'):
            events=torch.load(directory/arm/'rounds.pt',weights_only=True)
            previous=initial
            for event in events:
                torch.testing.assert_close(event['before'],previous,rtol=0,atol=0)
                if not event['record']['accepted']:
                    torch.testing.assert_close(event['before'],event['after'],rtol=0,atol=0)
                b=event['boundaries']
                if b is not None:
                    assert len(b['input_indices'])==len(b['points'])
                    assert (b['input_indices']>=0).all() and (b['input_indices']<len(event['target_ids'])).all()
                previous=event['after'];rounds+=1
            if reference is None:reference=events
            else:
                for left,right in zip(reference,events):
                    assert torch.equal(left['source_ids'],right['source_ids'])
                    assert torch.equal(left['target_ids'],right['target_ids'])
                    paired+=1
            model.load_state_dict(torch.load(directory/arm/'classifier.pt',weights_only=True))
            torch.testing.assert_close(params.vector(),previous,rtol=0,atol=0)
            if arm=='zero':torch.testing.assert_close(params.vector(),initial,rtol=0,atol=0)
            with torch.no_grad():
                torch.testing.assert_close(model(inputs['evaluation']),torch.load(directory/arm/'predictions.pt',weights_only=True),rtol=0,atol=0)
            checkpoints+=1
        score_data=torch.load(directory/'score.pt',weights_only=True)
        score=ScoreNetwork(dim,score_data['hidden']).to(dtype=source.dtype)
        score.load_state_dict(score_data['state_dict']);score.eval()
        banks=torch.load(directory/'queries.pt',weights_only=True)
        with torch.no_grad():
            for bank in banks.values():
                if len(bank['points']):torch.testing.assert_close(score(bank['points']),bank['score_learned'],rtol=0,atol=0)
        score_checkpoints+=1
    return dict(files_verified=len(manifest),final_classifier_checkpoints_replayed=checkpoints,
                score_checkpoints_replayed=score_checkpoints,round_state_chains_verified=rounds,
                paired_round_id_comparisons=paired,prediction_replay_atol=0,prediction_replay_rtol=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    result=verify(args.run.resolve());write_json(args.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
