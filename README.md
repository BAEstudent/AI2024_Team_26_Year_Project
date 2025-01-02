># AI2024_Team_26_Year_Project
>Repository for team 26 first-year final project solution.

>## Project: Processing and Analysis of Medical Images
>Goal: an app that classifies medical conditions based on images of pigmented skin lesions [training set taken from link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T&version=4.0)

>## Curator
>- Блуменау Марк (@markblumenau)

>## Team members
>- Карасев Дмитрий Алексеевич (@dmitry_0123)
>- Палагин Иван Дмитриевич (@n0gr1m)
>- Сидельников Платон Павлович (@platonsidelnikov)
>- Юдайкин Кирилл Вячеславович (@Kirill_Yudaykin)


>## How to start docker template
>Для сборки образа и контейнеров необходимо перейти в папку Checkpoint_04 и запустить docker compose up —build. После чего контейнеры с API/Streamlit -приложением будут запушены.



>## Demonstration of MVP work
>### Uploading dataset
> Первым делом необходимо загрузить датасет в формате .zip.
>![](assets/fit.gif)
>### EDA view
> Чтобы посмотреть EDA, необходимо:
>- нажать на кнопку “Показать EDA”;
>- выбрать по каким данным: метаданные или картинки.
> После просмотра Вы можете скрыть EDA, нажав на кнопку “Скрыть EDA”.
>![](assets/EDA.gif)
>### Creating new model
> Для создания новой модели необходимо:
>- ввести название модели в специальном окне;
>- указать гиперпараметры обучения;
>- нажать на кнопку "Создать и обучить модель".
>![](assets/learn_full.gif)
>### Getting information about created models
> Получить информацию о гиперпараметрах обученной модели, а также посмотреть на ее кривую обучения можно, нажав кнопку “Получить информацию о моделях”.
>![](assets/stats.gif)
>### Inference
> Чтобы произвести инференс на обученной модели, необходимо:
>- нажать на кнопку “Начать инференс”;
>- выбрать нужную модель из выпадающего списка;
>- загрузить фотография в формате .jpg.
>![](assets/inference.gif)
>### Deleting created models
> Чтобы очистить список обученных моделей, нажмите на кнопку “Удалить все Ваши модели”.
>![](assets/delete.gif)
