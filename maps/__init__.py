class BaseMapScenario:
    """
    Base class for all TacBench map scenarios.

    Subclass this in maps/<map_name>.py and override the methods below.
    The loader instantiates MapScenario() from the matching file, so the
    class must be named exactly MapScenario in each map file.

    settings_overrides is reserved for future per-map settings support —
    populate it as a dict of TACBENCH_ field names to values and the
    loader will apply them after the scenario is loaded.
    """

    briefing: str = ""
    settings_overrides: dict = {}

    def on_step(self, bot) -> None:
        """Update internal scenario state. Called every game step before win/loss checks."""
        pass

    def check_win(self, bot) -> bool:
        """Return True if the win condition has been met."""
        return False

    def check_loss(self, bot) -> bool:
        """Return True if the loss condition has been met."""
        return False
