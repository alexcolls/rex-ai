
import pandas as pd
pd.options.plotting.backend = "plotly"

def plotWeek( year=2022, week=1 ):

    path = 'db/data/secondary/'+str(year)+'/'+str(week)+'/'

    # load mid prices
    mids_ = pd.read_csv(path+'mids_.csv', index_col=0)

    # plot portfolio returns
    mids_.plot(title=f'Weekly market returns (%) {year}, week {week}').show()

    idxs_ = pd.read_csv(path+'idxs_.csv', index_col=0)

    # plot portfolio returns
    idxs_.plot(title=f'Weekly currency returns (%) {year}, week {week}').show()