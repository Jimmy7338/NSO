#!/usr/bin/env python3
"""Record strict-frontier dispatch and reach receipts through the native driver."""

import runner


def attach_frontier_receipts(driver, runtime):
    choose = driver.next_action
    observe = runtime.observe
    selected = {}
    reached = set()
    active = None

    def action():
        nonlocal active
        command = choose()
        state = runtime.states[0]
        decision = driver.last_decision
        active = None
        if command is None or state['phase'] != 'outbound' or decision is None:
            return command
        goal=decision.get('goal')
        if goal is None:
            return command
        heading=driver.policy.goal_heading
        count=driver.policy.goal_count
        if heading is None:
            return command
        active=count
        target=[*goal,int(heading)]
        if count not in selected:
            selected[count]=target
        driver.events.append(dict(event='instrumented_frontier_dispatch',paid_before=state['packet'].action_id,
             goal_count=count,goal_pose=target,action=command,guard=state['pending']))
        return command

    def observation(scene,*args,**kwargs):
        value=observe(scene,*args,**kwargs)
        packet=runtime.states[scene]['packet']
        current=[*packet.position,packet.heading]
        if active is not None and active not in reached and current==selected[active]:
            reached.add(active)
            driver.events.append(dict(event='instrumented_frontier_observation',paid=packet.action_id,
                                      goal_count=active,goal_reached=True,actual_pose=current))
        return value

    driver.next_action=action
    runtime.observe=observation
    return driver


def start():
    original_import=runner.import_file

    def import_bound(name,path):
        module=original_import(name,path)
        if name=='original_weighted_actor':
            original_frontier=module.native.frontier_adapter

            def frontier(components,runtime,world):
                return attach_frontier_receipts(original_frontier(components,runtime,world),runtime)

            module.native.frontier_adapter=frontier
        return module

    runner.import_file=import_bound
    runner.main()


if __name__=='__main__':
    start()
