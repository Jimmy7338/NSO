#!/usr/bin/env python3
"""Count distinct recorded global-choice events and repeated next-action returns."""
import collections,json,pathlib,sys
rows=[json.loads(s) for s in pathlib.Path(sys.argv[1]).read_text().splitlines()];calls=[x['response']['result'] for x in rows if x['request']['method']=='next_action' and x['response'].get('ok')];plans={};events=collections.Counter()
for result in calls:
 for e in result.get('runtime_events',[]):
  if isinstance(e,dict):events[e.get('event')]+=1
 selection=result.get('selection') or {};selected=result.get('selected') or {};call=selected.get('selection_call_id')
 if call is None:continue
 scores=selection.get('scores') or {};g=scores.get('G',[]);s=scores.get('S',[]);candidates=selection.get('candidates',[]);assert len(g)==len(s)==len(candidates) and g
 row={'selection_call_id':call,'plan_number':selection.get('plan_number'),'selected_map_version':selected.get('selected_map_version'),'candidate_count':len(g),'G_equals_S_per_candidate':g==s,'selected_candidate':selected.get('candidate_id')}
 if call in plans:assert plans[call]['selection']==row
 else:plans[call]={'selection':row,'next_action_returns':0,'global_choice_events':0}
 plans[call]['next_action_returns']+=1;plans[call]['global_choice_events']+=sum(e.get('event')=='global_choice' for e in result.get('runtime_events',[]) if isinstance(e,dict))
plan_list=list(plans.values());assert [x['selection']['plan_number'] for x in plan_list]==list(range(1,len(plan_list)+1));assert all(x['global_choice_events']==1 for x in plan_list);assert events['global_choice']==len(plan_list)==calls[-1]['summary']['runtime']['modules']['plans']
print(json.dumps({'schema':'nso.recorded_selection_count.v1','next_action_returns':len(calls),'global_choice_events':events['global_choice'],'final_runtime_plan_count':calls[-1]['summary']['runtime']['modules']['plans'],'plans':plan_list,'limits':['Repeated next_action responses are not independent plans','Equality is per candidate at recorded states, not matched physical efficacy','Ten later plans have only one candidate']},indent=2))
