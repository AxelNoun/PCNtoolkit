Compare normative models
========================

.. container:: notebook-download

   :download:`Download Jupyter notebook <notebooks/07_model_comparison.ipynb>`

This tutorial fits two HBR models on the same data and asks which one
predicts better.

- ``model1`` has a single intercept for the mean, shared by all sites
  and sexes.
- ``model2`` gives each site and each sex its own intercept (a random
  intercept).

Everything else is identical, so the comparison tells us whether letting
the mean differ between sites and sexes is worth the extra parameters.

.. code:: ipython3

    import logging
    import warnings
    
    import seaborn as sns
    
    import pcntoolkit.util.output
    from pcntoolkit import (
        HBR,
        MCMC,
        BsplineBasisFunction,
        NormalLikelihood,
        NormativeModel,
        NormData,
        load_fcon1000,
        make_prior,
    )
    from pcntoolkit.util.model_comparison import compare_hbr_models
    
    sns.set_style("darkgrid")
    
    # Suppress some annoying warnings and logs
    pymc_logger = logging.getLogger("pymc")
    
    pymc_logger.setLevel(logging.WARNING)
    pymc_logger.propagate = False
    
    warnings.simplefilter(action="ignore", category=FutureWarning)
    # Show warnings without the path of the file that raised them
    warnings.formatwarning = lambda message, category, *args, **kwargs: (
        f"{category.__name__}: {message}\n"
    )
    pcntoolkit.util.output.Output.set_show_messages(False)

.. code:: ipython3

    # Download an example dataset
    norm_data: NormData = load_fcon1000()
    
    # Select only a few features
    features_to_model = [
        "WM-hypointensities",
        "Right-Lateral-Ventricle",
        # "Right-Amygdala",
        # "CortexVol",
    ]
    norm_data = norm_data.sel({"response_vars": features_to_model})
    
    # Split into train and test sets
    train, test = norm_data.train_test_split()

.. code:: ipython3

    mu1 = make_prior(
        # Mu is linear because we want to allow the mean to vary as a function of the covariates.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 10.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 5.0)),
        # The intercept is not random, because we want to compare to a model with random intercept
        intercept=make_prior(
            dist_name="Normal",
            dist_params=(0.0, 2.0),
        ),
        # We use a B-spline basis function to allow for non-linearity in the mean.
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )
    sigma1 = make_prior(
        # Sigma is also linear, because we want to allow the standard deviation to vary as a function of the covariates: heteroskedasticity.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 2.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
        # The intercept is not random, because we assume the intercept of the variance to be the same for all sites and sexes.
        intercept=make_prior(dist_name="Normal", dist_params=(1.0, 1.0)),
        # We use a B-spline basis function to allow for non-linearity in the standard deviation.
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
        # We use a softplus mapping to ensure that sigma is strictly positive.
        mapping="softplus",
        # We scale the softplus mapping by a factor of 3, to avoid spikes in the resulting density.
        # The parameters (a, b, c) provided to a mapping f are used as: f_abc(x) = f((x - a) / b) * b + c
        # This basically provides an affine transformation of the softplus function.
        # a -> horizontal shift
        # b -> scaling
        # c -> vertical shift
        # You can leave c out, and it will default to 0.
        mapping_params=(0.0, 3.0),
    )
    # Set the likelihood with the priors we just created.
    likelihood1 = NormalLikelihood(mu1, sigma1)
    
    template_hbr_1 = HBR(
        name="template",
        # The likelihood function to use for the model.
        likelihood=likelihood1,
        # How to estimate the model parameters: here MCMC sampling.
        inference_method=MCMC(
            # The number of cores to use for sampling.
            cores=16,
            # Whether to show a progress bar during the model fitting.
            progressbar=False,
            # The number of draws to sample from the posterior per chain.
            draws=1500,
            # The number of tuning steps to run.
            tune=500,
            # The number of MCMC chains to run.
            chains=4,
            # The sampler to use for the model.
            nuts_sampler="nutpie",
        ),
    )
    model1 = NormativeModel(
        # The regression model to use for the normative model.
        template_regression_model=template_hbr_1,
        # Whether to save the model after fitting.
        savemodel=True,
        # Whether to evaluate the model after fitting.
        evaluate_model=True,
        # Whether to save the results after evaluation.
        saveresults=True,
        # Whether to save the plots after fitting.
        saveplots=False,
        # The directory to save the model, results, and plots.
        save_dir="resources/compare_hbr/model1",
        # The scaler to use for the input data. Can be either one of "standardize", "minmax", "robminmax", "none"
        inscaler="standardize",
        # The scaler to use for the output data. Can be either one of "standardize", "minmax", "robminmax", "none"
        outscaler="standardize",
    )

.. code:: ipython3

    mu2 = make_prior(
        # Mu is linear because we want to allow the mean to vary as a function of the covariates.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 10.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 5.0)),
        # The intercept is random, because we expect the intercept to vary between sites and sexes.
        intercept=make_prior(
            random=True,
            # Mu is the mean of the intercept, which is normally distributed with a mean of 0 and a standard deviation of 1.
            mu=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
            # Sigma is the scale at which the intercepts vary. It is a positive parameter, so we have to map it to the positive domain.
            sigma=make_prior(
                dist_name="Normal",
                dist_params=(1.0, 0.5),
                mapping="softplus",
                mapping_params=(0.0, 2.0),
            ),
        ),
        # We use a B-spline basis function to allow for non-linearity in the mean.
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )
    sigma2 = make_prior(
        # Sigma is also linear, because we want to allow the standard deviation to vary as a function of the covariates: heteroskedasticity.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 2.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
        # The intercept is not random, because we assume the intercept of the variance to be the same for all sites and sexes.
        intercept=make_prior(dist_name="Normal", dist_params=(1.0, 1.0)),
        # We use a B-spline basis function to allow for non-linearity in the standard deviation.
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
        # We use a softplus mapping to ensure that sigma is strictly positive.
        mapping="softplus",
        # We scale the softplus mapping by a factor of 3, to avoid spikes in the resulting density.
        # The parameters (a, b, c) provided to a mapping f are used as: f_abc(x) = f((x - a) / b) * b + c
        # This basically provides an affine transformation of the softplus function.
        # a -> horizontal shift
        # b -> scaling
        # c -> vertical shift
        # You can leave c out, and it will default to 0.
        mapping_params=(0.0, 3.0),
    )
    # Set the likelihood with the priors we just created.
    likelihood2 = NormalLikelihood(mu2, sigma2)
    
    template_hbr_2 = HBR(
        name="template",
        # The likelihood function to use for the model.
        likelihood=likelihood2,
        # How to estimate the model parameters: here MCMC sampling.
        inference_method=MCMC(
            # The number of cores to use for sampling.
            cores=16,
            # Whether to show a progress bar during the model fitting.
            progressbar=False,
            # The number of draws to sample from the posterior per chain.
            draws=1500,
            # The number of tuning steps to run.
            tune=500,
            # The number of MCMC chains to run.
            chains=4,
            # The sampler to use for the model.
            nuts_sampler="nutpie",
        ),
    )
    model2 = NormativeModel(
        # The regression model to use for the normative model.
        template_regression_model=template_hbr_2,
        # Whether to save the model after fitting.
        savemodel=True,
        # Whether to evaluate the model after fitting.
        evaluate_model=True,
        # Whether to save the results after evaluation.
        saveresults=True,
        # Whether to save the plots after fitting.
        saveplots=False,
        # The directory to save the model, results, and plots.
        save_dir="resources/compare_hbr/model2",
        # The scaler to use for the input data. Can be either one of "standardize", "minmax", "robminmax", "none"
        inscaler="standardize",
        # The scaler to use for the output data. Can be either one of "standardize", "minmax", "robminmax", "none"
        outscaler="standardize",
    )

.. code:: ipython3

    model1.fit_predict(train, test);
    model2.fit_predict(train, test);

.. code:: ipython3

    # Delete references to model objects to ensure what follows will work for models saved to disk too
    del model1
    del model2

Compare the models
------------------

``compare_hbr_models`` scores each model by its ELPD (expected log
pointwise predictive density). For every subject, it takes the log of
the predictive density the model gives to that subject’s value when the
subject is left out of the fit, and sums over subjects. A model that
puts more density on what is actually observed gets a higher ELPD, so
higher is better.

Refitting the model once per left-out subject would be far too slow. The
ELPD is instead estimated from the posterior samples we already have
(the parameter values the sampler drew during the fit), with PSIS-LOO
(Pareto-smoothed importance sampling leave-one-out cross-validation),
which reweights the samples to mimic the removal of each subject.

This is why we pass ``train`` and not ``test``. A subject can only be
left out of a fit it was part of. The models never saw the test
subjects, so reweighting the samples to remove them would not give a
leave-one-out estimate. And because the score already describes subjects
the model has not seen, the comparison does not need a separate test
set.

.. code:: ipython3

    dct = {
        "model1": "resources/compare_hbr/model1",
        "model2": "resources/compare_hbr/model2",
    }
    # The models are scored on the data they were fitted on
    comparison = compare_hbr_models(dct, train)


.. code:: text

    UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
    UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
    UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
    UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.


.. code:: ipython3

    for k, v in comparison.items():
        print(k)
        display(v)


.. code:: text

    WM-hypointensities



.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>rank</th>
          <th>elpd_diff</th>
          <th>dse</th>
          <th>p_worse</th>
          <th>diag_diff</th>
          <th>diag_elpd</th>
          <th>p</th>
          <th>elpd</th>
          <th>se</th>
          <th>weight</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>model2</th>
          <td>0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>NaN</td>
          <td></td>
          <td>5 k̂ &gt; 0.70</td>
          <td>47.8</td>
          <td>-780.0</td>
          <td>46.0</td>
          <td>0.98</td>
        </tr>
        <tr>
          <th>model1</th>
          <td>1</td>
          <td>-90.0</td>
          <td>13.0</td>
          <td>1.0</td>
          <td></td>
          <td>2 k̂ &gt; 0.70</td>
          <td>27.5</td>
          <td>-870.0</td>
          <td>43.0</td>
          <td>0.02</td>
        </tr>
      </tbody>
    </table>
    </div>


.. code:: text

    Right-Lateral-Ventricle



.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>rank</th>
          <th>elpd_diff</th>
          <th>dse</th>
          <th>p_worse</th>
          <th>diag_diff</th>
          <th>diag_elpd</th>
          <th>p</th>
          <th>elpd</th>
          <th>se</th>
          <th>weight</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>model2</th>
          <td>0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>NaN</td>
          <td></td>
          <td>4 k̂ &gt; 0.70</td>
          <td>47.8</td>
          <td>-1090.0</td>
          <td>45.0</td>
          <td>0.6</td>
        </tr>
        <tr>
          <th>model1</th>
          <td>1</td>
          <td>-0.0</td>
          <td>8.5</td>
          <td>0.69</td>
          <td></td>
          <td>1 k̂ &gt; 0.70</td>
          <td>21.5</td>
          <td>-1100.0</td>
          <td>46.0</td>
          <td>0.4</td>
        </tr>
      </tbody>
    </table>
    </div>


Reading the tables
------------------

There is one table per response variable, with the best model on top.
The columns to look at first:

- ``elpd``: the score. Its absolute value means little; what counts is
  the difference between models fitted on the same data.
- ``elpd_diff``: the gap to the best model, which is 0 for the best
  model itself.
- ``dse``: the standard error of that gap. A gap several times larger
  than ``dse`` is a clear difference. A gap smaller than about 4, or of
  the same size as ``dse``, is not (``diag_diff`` flags the first case).
- ``p``: the effective number of parameters, which measures how flexible
  the model is.
- ``weight``: how much each model would contribute if their predictions
  were averaged.
- ``diag_elpd``: the number of subjects for which the leave-one-out
  estimate is unreliable. The warning above is about these subjects.

MCMC sampling is random and its seed is not fixed here, so your numbers
will differ slightly from the ones shown.

For ``WM-hypointensities``, ``model2`` wins clearly. ``model1`` is about
90 ELPD units behind, which is several times ``dse``: letting the mean
differ between sites and sexes improves the predictions for this
measure.

For ``Right-Lateral-Ventricle``, the two models cannot be told apart.
The gap is smaller than ``dse``, although ``model2`` uses more than
twice as many effective parameters (``p``). The random intercept does
not clearly improve the predictions here, so the simpler ``model1`` does
about as well.
