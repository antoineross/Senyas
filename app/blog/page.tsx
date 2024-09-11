import { BlogPosts } from 'app/components/posts'

export const metadata = {
  title: 'Senyas FSL Translator Blog',
  description: 'The authors of Senyas write about their experiences and insights after working on a web application that translates FSL into text through computer vision and machine learning.',
}

export default function Page() {
  return (
    <section>
      <h1 className="font-semibold text-2xl mb-8 tracking-tighter">My Blog</h1>
      <BlogPosts />
    </section>
  )
}
