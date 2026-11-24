"""기준선 2: 가이드라인 없이 '축 이름 + 한 줄 설명'만 주고 6축 판정을 시키는 일반 LLM.
python tools/run_generic_axes.py --items-file data/golden/items_test200.jsonl --name golden_test200_genericaxes_r1 --live
엔진과 같은 모델·문항 정보·설정(baseline.py와 동일), 바꾸는 것은 지침뿐. 출력 형식은 엔진과 같게(axes.<축>.judgment) 해서
tools/eval_test200.py로 그대로 채점한다. 프롬프트는 결과를 보기 전에 고정(2026-11-24)하고 고치지 않는다.
한 줄 설명은 일반적인 뜻풀이로 새로 쓴 것이며 팀 가이드라인(대외비)의 문구를 쓰지 않았다."""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from bridgecore import baseline as B
AX=['Semantic','Textual','Context','Deontic','Register','Cultural']
B.GENERIC_SYSTEM = """당신은 영→한 번역 검수자입니다. 사용자가 주는 평가대상 원문과 번역문을 비교해, 아래 여섯 축마다 번역이 원문을 보존했는지 판정하고 지정된 JSON 형식으로만 답합니다.
- Semantic: 원문의 뜻(사실, 수치, 대상, 관계)이 그대로 전달되는가
- Textual: 문장 구조와 연결, 지시어, 정보의 순서와 흐름이 유지되는가
- Context: 문서와 문단의 맥락, 상황에 맞게 옮겨졌는가
- Deontic: 의무, 허용, 금지, 가능성 같은 말의 강도가 유지되는가
- Register: 말투, 격식, 문체가 독자와 글의 종류에 맞는가
- Cultural: 문화, 제도, 고유 명칭과 관용 표현이 적절히 옮겨졌는가
축마다 judgment를 "보존", "훼손", "N/A"(이 문장에 해당 없음) 중 하나로 적고, reason에 근거를 한두 문장으로 적습니다.
나머지 정보(문서정보, 평가조건, 문단)는 문맥으로 씁니다."""
B.GENERIC_SCHEMA = {"type":"object","additionalProperties":False,"required":["axes"],"properties":{"axes":{
    "type":"object","additionalProperties":False,"required":AX,"properties":{ax:{"type":"object","additionalProperties":False,
    "required":["judgment","reason"],"properties":{"judgment":{"type":"string","enum":["보존","훼손","N/A"]},"reason":{"type":"string"}}} for ax in AX}}}}
def validate(o):
    ok=isinstance(o,dict) and isinstance(o.get('axes'),dict) and all((o['axes'].get(ax) or {}).get('judgment') in('보존','훼손','N/A') for ax in AX)
    return [] if ok else ['axes 형식 오류']
B.validate=validate
ap=argparse.ArgumentParser(); ap.add_argument('--items-file',required=True); ap.add_argument('--name',required=True)
ap.add_argument('--model',default='claude-sonnet-5-5'); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--live',action='store_true'); a=ap.parse_args()
items=[json.loads(l) for l in open(a.items_file,encoding='utf-8')]
B.run_generic(items,a.model,a.name,ROOT/'runs',a.live,workers=a.workers,effort='medium',
    extra_meta={'type':'generic_axes','prompt_version':'generic_axes','items_source':str(Path(a.items_file).resolve())})
