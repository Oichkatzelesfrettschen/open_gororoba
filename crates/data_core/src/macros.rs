/// Generate a [`DatasetProvider`](crate::fetcher::DatasetProvider) impl for
/// providers that follow the common
/// "download_with_fallbacks → single output file" pattern.
///
/// Eliminates a latent bug class: it is impossible for `fetch()` and
/// `is_cached()` to disagree on the output filename when both are
/// generated from the same `$output_file` token.
///
/// # Example
///
/// ```ignore
/// simple_provider! {
///     /// CHIME Catalog 1 dataset provider.
///     pub struct ChimeCat1Provider;
///     name = "CHIME/FRB Catalog 1";
///     output = "chime_frb_cat1.csv";
///     urls = CATALOG1_URLS;
/// }
/// ```
macro_rules! simple_provider {
    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident;
        name = $display_name:expr;
        output = $output_file:expr;
        urls = $urls:expr;
    ) => {
        $(#[$attr])*
        $vis struct $name;

        impl $crate::fetcher::DatasetProvider for $name {
            fn name(&self) -> &str {
                $display_name
            }

            fn fetch(
                &self,
                config: &$crate::fetcher::FetchConfig,
            ) -> ::std::result::Result<::std::path::PathBuf, $crate::fetcher::FetchError> {
                let output = config.output_dir.join($output_file);
                $crate::fetcher::download_with_fallbacks(
                    self.name(),
                    $urls,
                    &output,
                    config.skip_existing,
                )
            }

            fn is_cached(&self, config: &$crate::fetcher::FetchConfig) -> bool {
                config.output_dir.join($output_file).exists()
            }
        }
    };
}
