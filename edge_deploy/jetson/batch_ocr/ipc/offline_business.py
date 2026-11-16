"""Project-local offline factory; never change installed Ultralytics settings."""


def load_business():
    from serve_rec_wide import original_factory, selected_config, load_models
    wide=selected_config()
    base=original_factory()
    server,root,config=base.deployment()
    # The original deployment establishes model-import compatibility/config paths first.
    from ultralytics.hub.utils import events
    events.enabled=False
    events.events.clear()
    pipeline,models=load_models(base,server,root,config,wide)
    return server,pipeline,models,config
